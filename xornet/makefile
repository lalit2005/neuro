CFLAGS = -Wall -Wextra -g

compile:
	gcc $(CFLAGS) main.c -o xornet -lm

execute:
	./xornet

clean:
	rm xornet

run: compile execute clean

.PHONY: compile execute clean run
