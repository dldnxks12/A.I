#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <termios.h> // for serial communication
#include <fcntl.h>

#define BUF_MAX 512

int main(void){

	int fd, i = 0; // fd : file descriptor
	char buf[BUF_MAX];
	char tmp, read_byte=0;

	struct termios newtio;

	// O_RDWR   : Read + Write
	// O_NOCTTY : Some flag aboout terminal control...
	fd = open("/dev/pts/4", O_RDWR | O_NOCTTY);
	if(fd < 0){ // file open fail ...
		fprintf(stderr, "ERROR\n");
		exit(-1); // process end
	}

	memset(&newtio, 0, sizeof(newtio)); // Assign dynamic memory

	// c_iflag : input mode
	// c_oflag : output mode
	// c_cflag : control mode
	// c_lflag : local mode
	// cc_t    : special characters

	// control mode
	newtio.c_cflag  = B115200; // Boad Rate
	newtio.c_cflag |= CS8;
	newtio.c_cflag |= CLOCAL;
	newtio.c_cflag |= CREAD;

	// input mode
	newtio.c_iflag = IGNPAR; // Ignore parity bit (No parity check)
	// newtio.c_iflag = ICRNL // Carriage Return

	newtio.c_oflag = 0;
	newtio.c_lflag = 0;

	newtio.c_cc[VTIME] = 0; // Timeout in deciseconds
	newtio.c_cc[VMIN]  = 0; // Minumum number of characters

	// transmit info to file descriptor
	tcflush(fd, TCIFLUSH);
	tcsetattr(fd, TCSANOW, &newtio); // TCSANOW : change occurs immediately


	sleep(3);

	while(1){

		write(fd, "R", 1);
		// i = read(fd, buf, BUF_MAX);
		// printf("%s", buf);
		sleep(1);
	}

	close(fd);
	return 0;

}


