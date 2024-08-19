# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang

#CC = gcc-14 -Wall
#CC = clang -Wall
CC = gcc -Wall

test: tknz
	./tknz "Humpty Dumpty sat on a wall, Humpty Dumpty had a great fall. All the king's horses and all the king's men Couldn't put Humpty together again."

test2: tknz
	./tknz "Mary had a little lamb, Little lamb, little lamb, Mary had a little lamb, It's fleece was white as snow. And everywhere that Mary went Mary went, Mary went Everywhere that Mary went The lamb was sure to go. It followed her to school one day school one day, school one day, It followed her to school one day That was against the rule."

tknz: tknz.c
	$(CC) tknz.c -o tknz -lm

clean:
	rm -f tknz

c: clean

