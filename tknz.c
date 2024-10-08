/* Tokenizer in pure C */
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#define VOCAB_SIZE 32000

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

void dump_tokenizer(Tokenizer* t, int vocab_size) {
  int n = t->vocab_size;
  for (int i=0; i < n; i++) {
    char* str = t->vocab[i];
    printf("%d: %s\n", i, str);
  }
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

char* decode_simple(Tokenizer* t, int prev_token, int token) {
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

char *decode(Tokenizer *t, int prev_token, int token) {
    // 通常の語彙からトークンを取得
    char *piece = t->vocab[token];

    // バイトトークンの場合の処理
    unsigned char byte_val;
    static char utf8_buffer[5]; // 最大4バイトのUTF-8コードポイント + 終端文字
    static int utf8_index = 0;  // 現在のバッファ内の位置

    // フォールバックトークンの場合
    if (token >= 3 && token <= 258) { // フォールバックで生成されたバイトトークンを判定
        byte_val = (unsigned char)(token - 3); // トークンIDから元のバイト値を復元

        // バッファにバイトを追加
        utf8_buffer[utf8_index++] = byte_val;

        // 完全なUTF-8コードポイントが揃ったかどうかの判定
        if ((utf8_buffer[0] & 0x80) == 0) {
            // 1バイトASCII文字 (0xxxxxxx) - すでにこれで一文字が完成
            utf8_buffer[utf8_index] = '\0'; // 終端を追加
            utf8_index = 0; // 次の文字に備えてリセット
            return utf8_buffer;
        } else if ((utf8_buffer[0] & 0xE0) == 0xC0 && utf8_index == 2) {
            // 2バイトUTF-8文字 (110xxxxx 10xxxxxx)
            utf8_buffer[utf8_index] = '\0'; // 終端を追加
            utf8_index = 0; // 次の文字に備えてリセット
            return utf8_buffer;
        } else if ((utf8_buffer[0] & 0xF0) == 0xE0 && utf8_index == 3) {
            // 3バイトUTF-8文字 (1110xxxx 10xxxxxx 10xxxxxx)
            utf8_buffer[utf8_index] = '\0'; // 終端を追加
            utf8_index = 0; // 次の文字に備えてリセット
            return utf8_buffer;
        } else if ((utf8_buffer[0] & 0xF8) == 0xF0 && utf8_index == 4) {
            // 4バイトUTF-8文字 (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
            utf8_buffer[utf8_index] = '\0'; // 終端を追加
            utf8_index = 0; // 次の文字に備えてリセット
            return utf8_buffer;
        } else {
            // まだ完全なUTF-8コードポイントに達していないので何も返さない
            return NULL;
        }
    }

    // それ以外の場合（通常のトークン）
    return piece;
}

char* safe_piece(char *piece) {
  // piece might be a raw byte token, and we only want to print printable chars or whitespace
  // because some of the other bytes can be various control codes, backspace, etc.
  char* safe_empty = ""; 
  if (piece == NULL) { return safe_empty; }
  if (piece[0] == '\0') { return safe_empty; }
  if (piece[1] == '\0') {
    unsigned char byte_val = piece[0];
    if (!(isprint(byte_val) || isspace(byte_val))) {
      return safe_empty; // bad byte, don't print it
    }
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

void generate(Tokenizer *tokenizer, char *prompt) {
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

  for (int i=1; i<num_prompt_tokens; i++) {
    int token = prompt_tokens[0];
    char* piece = decode(tokenizer, token, prompt_tokens[i]);
    printf ("[% 3d]:% 6d = '%s'\n", i, prompt_tokens[i], safe_piece(piece));
  }

  printf("%d tokens.\n", num_prompt_tokens-1);
  fflush(stdout);
  free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
  fprintf(stderr, "Usage:   run \"<input tokens>\"\n");
  fprintf(stderr, "Example: run \"Once upon a time\"\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
  char *tokenizer_path = "tokenizer.bin";
  char *prompt = NULL;
  if (argc >= 2) { prompt = argv[1]; } else { error_usage(); }
  Tokenizer tokenizer;
  build_tokenizer(&tokenizer, tokenizer_path, VOCAB_SIZE);

  if (0) dump_tokenizer(&tokenizer, VOCAB_SIZE);

  fflush(stdout);
  generate(&tokenizer, prompt);
  free_tokenizer(&tokenizer);
  return 0;
}
#endif
