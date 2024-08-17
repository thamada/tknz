/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
  #include "win.h"
#else
  #include <unistd.h>
  #include <sys/mman.h>
#endif
// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
  int dim; // transformer dimension
  int hidden_dim; // for ffn layers
  int n_layers; // number of layers
  int n_heads; // number of query heads
  int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
  int vocab_size; // vocabulary size, usually 256 (byte-level)
  int seq_len; // max sequence length
} Config;

typedef struct {
  // token embedding table
  float* token_embedding_table;    // (vocab_size, dim)
  // weights for rmsnorms
  float* rms_att_weight; // (layer, dim) rmsnorm weights
  float* rms_ffn_weight; // (layer, dim)
  // weights for matmuls. note dim == n_heads * head_size
  float* wq; // (layer, dim, n_heads * head_size)
  float* wk; // (layer, dim, n_kv_heads * head_size)
  float* wv; // (layer, dim, n_kv_heads * head_size)
  float* wo; // (layer, n_heads * head_size, dim)
  // weights for ffn
  float* w1; // (layer, hidden_dim, dim)
  float* w2; // (layer, dim, hidden_dim)
  float* w3; // (layer, hidden_dim, dim)
  // final rmsnorm
  float* rms_final_weight; // (dim,)
  // (optional) classifier weights for the logits, on the last layer
  float* wcls;
} TransformerWeights;

typedef struct {
  // current wave of activations
  float *x; // activation at current time stamp (dim,)
  float *xb; // same, but inside a residual branch (dim,)
  float *xb2; // an additional buffer just for convenience (dim,)
  float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
  float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
  float *q; // query (dim,)
  float *k; // key (dim,)
  float *v; // value (dim,)
  float *att; // buffer for scores/attention values (n_heads, seq_len)
  float *logits; // output logits
  // kv cache
  float* key_cache;   // (layer, seq_len, dim)
  float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
  Config config; // the hyperparameters of the architecture (the blueprint)
  TransformerWeights weights; // the weights of the model
  RunState state; // buffers for the "wave" of activations in the forward pass
  // some more state needed to properly clean up the memory mapping (sigh)
  int fd; // file descriptor for memory mapping
  float* data; // memory mapped data pointer
  ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
  // we calloc instead of malloc to keep valgrind happy
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  s->x = calloc(p->dim, sizeof(float));
  s->xb = calloc(p->dim, sizeof(float));
  s->xb2 = calloc(p->dim, sizeof(float));
  s->hb = calloc(p->hidden_dim, sizeof(float));
  s->hb2 = calloc(p->hidden_dim, sizeof(float));
  s->q = calloc(p->dim, sizeof(float));
  s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
  s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
  s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
  s->logits = calloc(p->vocab_size, sizeof(float));
  // ensure all mallocs went fine
  if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
	  || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
	fprintf(stderr, "malloc failed!\n");
	exit(EXIT_FAILURE);
  }
}

void free_run_state(RunState* s) {
  free(s->x);
  free(s->xb);
  free(s->xb2);
  free(s->hb);
  free(s->hb2);
  free(s->q);
  free(s->att);
  free(s->logits);
  free(s->key_cache);
  free(s->value_cache);
}

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
  int head_size = p->dim / p->n_heads;
  // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
  unsigned long long n_layers = p->n_layers;
  w->token_embedding_table = ptr;
  ptr += p->vocab_size * p->dim;
  w->rms_att_weight = ptr;
  ptr += n_layers * p->dim;
  w->wq = ptr;
  ptr += n_layers * p->dim * (p->n_heads * head_size);
  w->wk = ptr;
  ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
  w->wv = ptr;
  ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
  w->wo = ptr;
  ptr += n_layers * (p->n_heads * head_size) * p->dim;
  w->rms_ffn_weight = ptr;
  ptr += n_layers * p->dim;
  w->w1 = ptr;
  ptr += n_layers * p->dim * p->hidden_dim;
  w->w2 = ptr;
  ptr += n_layers * p->hidden_dim * p->dim;
  w->w3 = ptr;
  ptr += n_layers * p->dim * p->hidden_dim;
  w->rms_final_weight = ptr;
  ptr += p->dim;
  ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
  ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
  w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
  FILE *file = fopen(checkpoint, "rb");
  if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
  // read in the config header
  if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
  // negative vocab size is hacky way of signaling unshared weights. bit yikes.
  int shared_weights = config->vocab_size > 0 ? 1 : 0;
  config->vocab_size = abs(config->vocab_size);
  // figure out the file size
  fseek(file, 0, SEEK_END); // move file pointer to end of file
  *file_size = ftell(file); // get the file size, in bytes
  fclose(file);
  // memory map the Transformer weights into the data pointer
  *fd = open(checkpoint, O_RDONLY); // open in read only mode
  if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
  *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
  if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
  float* weights_ptr = *data + sizeof(Config)/sizeof(float);
  memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
  // read in the Config and the Weights from the checkpoint
  read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
  // allocate the RunState buffers
  malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
  // close the memory mapping
  if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
  if (t->fd != -1) { close(t->fd); }
  // free the RunState buffers
  free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, float* x, float* weight, int size) {
  // calculate sum of squares
  float ss = 0.0f;
  for (int j = 0; j < size; j++) {
	ss += x[j] * x[j];
  }
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
  // normalize and scale
  for (int j = 0; j < size; j++) {
	o[j] = weight[j] * (ss * x[j]);
  }
}

void softmax(float* x, int size) {
  // find max value (for numerical stability)
  float max_val = x[0];
  for (int i = 1; i < size; i++) {
	if (x[i] > max_val) {
	  max_val = x[i];
	}
  }
  // exp and sum
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
	x[i] = expf(x[i] - max_val);
	sum += x[i];
  }
  // normalize
  for (int i = 0; i < size; i++) {
	x[i] /= sum;
  }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  int i;
#pragma omp parallel for private(i)
  for (i = 0; i < d; i++) {
	float val = 0.0f;
	for (int j = 0; j < n; j++) {
	  val += w[i * n + j] * x[j];
	}
	xout[i] = val;
  }
}

float* forward(Transformer* transformer, int token, int pos) {

  // a few convenience variables
  Config* p = &transformer->config;
  TransformerWeights* w = &transformer->weights;
  RunState* s = &transformer->state;
  float *x = s->x;
  int dim = p->dim;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
  int hidden_dim =  p->hidden_dim;
  int head_size = dim / p->n_heads;

  // copy the token embedding into x
  float* content_row = w->token_embedding_table + token * dim;
  memcpy(x, content_row, dim*sizeof(*x));

  // forward all the layers
  for(unsigned long long l = 0; l < p->n_layers; l++) {

	// attention rmsnorm
	rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

	// key and value point to the kv cache
	int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
	s->k = s->key_cache + loff + pos * kv_dim;
	s->v = s->value_cache + loff + pos * kv_dim;

	// qkv matmuls for this position
	matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
	matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
	matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

	// RoPE relative positional encoding: complex-valued rotate q and k in each head
	for (int i = 0; i < dim; i+=2) {
	  int head_dim = i % head_size;
	  float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
	  float val = pos * freq;
	  float fcr = cosf(val);
	  float fci = sinf(val);
	  int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
	  for (int v = 0; v < rotn; v++) {
		float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
		float v0 = vec[i];
		float v1 = vec[i+1];
		vec[i]   = v0 * fcr - v1 * fci;
		vec[i+1] = v0 * fci + v1 * fcr;
	  }
	}

	// multihead attention. iterate over all heads
	int h;
#pragma omp parallel for private(h)
	for (h = 0; h < p->n_heads; h++) {
	  // get the query vector for this head
	  float* q = s->q + h * head_size;
	  // attention scores for this head
	  float* att = s->att + h * p->seq_len;
	  // iterate over all timesteps, including the current one
	  for (int t = 0; t <= pos; t++) {
		// get the key vector for this head and at this timestep
		float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
		// calculate the attention score as the dot product of q and k
		float score = 0.0f;
		for (int i = 0; i < head_size; i++) {
		  score += q[i] * k[i];
		}
		score /= sqrtf(head_size);
		// save the score to the attention buffer
		att[t] = score;
	  }

	  // softmax the scores to get attention weights, from 0..pos inclusively
	  softmax(att, pos + 1);

	  // weighted sum of the values, store back into xb
	  float* xb = s->xb + h * head_size;
	  memset(xb, 0, head_size * sizeof(float));
	  for (int t = 0; t <= pos; t++) {
		// get the value vector for this head and at this timestep
		float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
		// get the attention weight for this timestep
		float a = att[t];
		// accumulate the weighted value into xb
		for (int i = 0; i < head_size; i++) {
		  xb[i] += a * v[i];
		}
	  }
	}

	// final matmul to get the output of the attention
	matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

	// residual connection back into x
	for (int i = 0; i < dim; i++) {
	  x[i] += s->xb2[i];
	}

	// ffn rmsnorm
	rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

	// Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
	// first calculate self.w1(x) and self.w3(x)
	matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
	matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

	// SwiGLU non-linearity
	for (int i = 0; i < hidden_dim; i++) {
	  float val = s->hb[i];
	  // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
	  val *= (1.0f / (1.0f + expf(-val)));
	  // elementwise multiply with w3(x)
	  val *= s->hb2[i];
	  s->hb[i] = val;
	}

	// final matmul to get the output of the ffn
	matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

	// residual connection
	for (int i = 0; i < dim; i++) {
	  x[i] += s->xb[i];
	}
  }

  // final rmsnorm
  rmsnorm(x, x, w->rms_final_weight, dim);

  // classifier into logits
  matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
  return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
  char *str;
  int id;
} TokenIndex;

typedef struct {
  char** vocab;
  float* vocab_scores;
  TokenIndex *sorted_vocab;
  int vocab_size;
  unsigned int max_token_length;
  unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
  return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
  // i should have written the vocab_size into the tokenizer file... sigh
  t->vocab_size = vocab_size;
  // malloc space to hold the scores and the strings
  t->vocab = (char**)malloc(vocab_size * sizeof(char*));
  t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
  t->sorted_vocab = NULL; // initialized lazily
  for (int i = 0; i < 256; i++) {
	t->byte_pieces[i * 2] = (unsigned char)i;
	t->byte_pieces[i * 2 + 1] = '\0';
  }
  // read in the file
  FILE *file = fopen(tokenizer_path, "rb");
  if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
  if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
  int len;
  for (int i = 0; i < vocab_size; i++) {
	if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
	if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
	t->vocab[i] = (char *)malloc(len + 1);
	if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
	t->vocab[i][len] = '\0'; // add the string terminating token
  }
  fclose(file);
}

void free_tokenizer(Tokenizer* t) {
  for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
  free(t->vocab);
  free(t->vocab_scores);
  free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
  char *piece = t->vocab[token];
  // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
  if (prev_token == 1 && piece[0] == ' ') { piece++; }
  // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
  // parse this and convert and return the actual byte
  unsigned char byte_val;
  if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
	piece = (char*)t->byte_pieces + byte_val * 2;
  }
  return piece;
}

void safe_printf(char *piece) {
  // piece might be a raw byte token, and we only want to print printable chars or whitespace
  // because some of the other bytes can be various control codes, backspace, etc.
  if (piece == NULL) { return; }
  if (piece[0] == '\0') { return; }
  if (piece[1] == '\0') {
	unsigned char byte_val = piece[0];
	if (!(isprint(byte_val) || isspace(byte_val))) {
	  return; // bad byte, don't print it
	}
  }
  printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
  // efficiently find the perfect match for str in vocab, return its index or -1 if not found
  TokenIndex tok = { .str = str }; // acts as the key to search for
  TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
  return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
  // encode the string text (input) into an upper-bound preallocated tokens[] array
  // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
  if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

  if (t->sorted_vocab == NULL) {
	// lazily malloc and sort the vocabulary
	t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
	for (int i = 0; i < t->vocab_size; i++) {
	  t->sorted_vocab[i].str = t->vocab[i];
	  t->sorted_vocab[i].id = i;
	}
	qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
  }

  // create a temporary buffer that will store merge candidates of always two consecutive tokens
  // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
  char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
  size_t str_len = 0;

  // start at 0 tokens
  *n_tokens = 0;

  // add optional BOS (=1) token, if desired
  if (bos) tokens[(*n_tokens)++] = 1;

  // add_dummy_prefix is true by default
  // so prepend a dummy prefix token to the input string, but only if text != ""
  // TODO: pretty sure this isn't correct in the general case but I don't have the
  // energy to read more of the sentencepiece code to figure out what it's doing
  if (text[0] != '\0') {
	int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
	tokens[(*n_tokens)++] = dummy_prefix;
  }

  // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
  // Code point ↔ UTF-8 conversion
  // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
  // U+0000	U+007F	    0xxxxxxx
  // U+0080	U+07FF	    110xxxxx	10xxxxxx
  // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
  // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

  // process the raw (UTF-8) byte sequence of the input string
  for (char *c = text; *c != '\0'; c++) {

	// reset buffer if the current byte is ASCII or a leading byte
	// 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
	// 0x80 is 10000000
	// in UTF-8, all continuation bytes start with "10" in first two bits
	// so in English this is: "if this byte is not a continuation byte"
	if ((*c & 0xC0) != 0x80) {
	  // this byte must be either a leading byte (11...) or an ASCII char (0x...)
	  // => reset our location, as we're starting a new UTF-8 codepoint
	  str_len = 0;
	}

	// append the current byte to the buffer
	str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
	str_buffer[str_len] = '\0';

	// while the next character is a continuation byte, continue appending
	// but if there are too many of them, just stop to avoid overruning str_buffer size.
	if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
	  continue;
	}

	// ok c+1 is not a continuation byte, so we've read in a full codepoint
	int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

	if (id != -1) {
	  // we found this codepoint in vocab, add it as a token
	  tokens[(*n_tokens)++] = id;
	} else {
	  // byte_fallback encoding: just encode each byte as a token
	  // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
	  // so the individual bytes only start at index 3
	  for (int i=0; i < str_len; i++) {
		tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
	  }
	}
	str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
  }

  // merge the best consecutive pair each iteration, according the scores in vocab_scores
  while (1) {
	float best_score = -1e10;
	int best_id = -1;
	int best_idx = -1;

	for (int i=0; i < (*n_tokens-1); i++) {
	  // check if we can merge the pair (tokens[i], tokens[i+1])
	  sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
	  int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
	  if (id != -1 && t->vocab_scores[id] > best_score) {
		// this merge pair exists in vocab! record its score and position
		best_score = t->vocab_scores[id];
		best_id = id;
		best_idx = i;
	  }
	}

	if (best_idx == -1) {
	  break; // we couldn't find any more pairs to merge, so we're done
	}

	// merge the consecutive pair (best_idx, best_idx+1) into new token best_id
	tokens[best_idx] = best_id;
	// delete token at position best_idx+1, shift the entire sequence back 1
	for (int i = best_idx+1; i < (*n_tokens-1); i++) {
	  tokens[i] = tokens[i+1];
	}
	(*n_tokens)--; // token length decreased
  }

  // add optional EOS (=2) token, if desired
  if (eos) tokens[(*n_tokens)++] = 2;

  free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
  float prob;
  int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
  int vocab_size;
  ProbIndex* probindex; // buffer used in top-p sampling
  float temperature;
  float topp;
  unsigned long long rng_state;
} Sampler;


int compare(const void* a, const void* b) {
  ProbIndex* a_ = (ProbIndex*) a;
  ProbIndex* b_ = (ProbIndex*) b;
  if (a_->prob > b_->prob) return -1;
  if (a_->prob < b_->prob) return 1;
  return 0;
}


void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
  sampler->vocab_size = vocab_size;
  sampler->temperature = temperature;
  sampler->topp = topp;
  sampler->rng_state = rng_seed;
  // buffer only used with nucleus sampling; may not need but it's ~small
  sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
  free(sampler->probindex);
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
  char *empty_prompt = "";
  if (prompt == NULL) { prompt = empty_prompt; }

  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS

  encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

  if (num_prompt_tokens < 1) {
	fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
	exit(EXIT_FAILURE);
  }

  printf("input tokens = %d\n", num_prompt_tokens);

  for (int i=0; i<num_prompt_tokens; i++) {
    printf ("%d:[%d]\n", i, prompt_tokens[i]);
  }

  printf("\n");
  fflush(stdout);

  free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
  fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
  fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
  fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
  fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
  fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
  fprintf(stderr, "  -i <string> input prompt\n");
  fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
  fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
  fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

  // default parameters
  char *checkpoint_path = NULL;  // e.g. out/model.bin
  char *tokenizer_path = "tokenizer.bin";
  float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
  float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
  int steps = 256;            // number of steps to run for
  char *prompt = NULL;        // prompt string
  unsigned long long rng_seed = 0; // seed rng with time by default
  char *mode = "generate";    // generate|chat
  char *system_prompt = NULL; // the (optional) system prompt to use in chat mode

  // poor man's C argparse so we can override the defaults above from the command line
  if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
  for (int i = 2; i < argc; i+=2) {
	// do some basic validation
	if (i + 1 >= argc) { error_usage(); } // must have arg after flag
	if (argv[i][0] != '-') { error_usage(); } // must start with dash
	if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
	// read in the args
	if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
	else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
	else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
	else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
	else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
	else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
	else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
	else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
	else { error_usage(); }
  }

  // parameter validation/overrides
  if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
  if (temperature < 0.0) temperature = 0.0;
  if (topp < 0.0 || 1.0 < topp) topp = 0.9;
  if (steps < 0) steps = 0;

  // build the Transformer via the model .bin file
  Transformer transformer;
  build_transformer(&transformer, checkpoint_path);
  if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length

  if(0){
	Config* c = &transformer.config;
	printf("dim = %d\n", c->dim);
	printf("hidden_dim = %d\n", c->hidden_dim);
	printf("n_layers = %d\n", c->n_layers);
	printf("n_heads = %d\n", c->n_heads);
	printf("n_kv_heads = %d\n", c->n_kv_heads);
	printf("vocab_size = %d\n", c->vocab_size);
	printf("seq_len = %d\n", c->seq_len);
	printf("-------------------\n");
	//exit(-1);
  }


  // build the Tokenizer via the tokenizer .bin file
  Tokenizer tokenizer;
  build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

  // build the Sampler
  Sampler sampler;
  build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

  // run!
  {
    printf("generate!\n\n");
    fflush(stdout);
    generate(&transformer, &tokenizer, &sampler, prompt, steps);
  }

  // memory and file handles cleanup
  free_sampler(&sampler);
  free_tokenizer(&tokenizer);
  free_transformer(&transformer);
  return 0;
}
#endif
