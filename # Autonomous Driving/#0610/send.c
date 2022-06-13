#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>  // write(), read(), close(), ...
#include <termios.h> // for serial communication - POSIX terminal control 
#include <fcntl.h>   // O_RDWR, ...
#include <sys/types.h>

#define BUF_MAX 1024

int main(void){

	// use read / write / open / close with this file descriptor
	int fd = 0; // fd : file descriptor
	char buf[BUF_MAX];

	// Open Log file
	FILE *pFile = NULL;
	pFile       = fopen("log.txt", "r");

	struct termios newtio;

	// O_RDWR   : Read + Write
	// O_NOCTTY : Some flag aboout terminal control...
	fd = open("buffer.txt", O_RDWR | O_NOCTTY);
	if(fd < 0){ // file open fail ...
		fprintf(stderr, "ERROR\n");
		exit(-1); // process end
	}
	
	memset(&newtio, 0, sizeof(newtio)); // Assign dynamic memory

	newtio.c_cflag  = B115200; // Boad Rate
	newtio.c_cflag |= CS8;     // 8bit , parity X , Stopbit 1
	newtio.c_cflag |= CLOCAL;  // Use internal port
	newtio.c_cflag |= CREAD;   // READ , WRITE OKAY

	// input mode
	newtio.c_iflag = IGNPAR; // Ignore parity bit (No parity check)
	//newtio.c_iflag = ICRNL; // Carriage Return

	newtio.c_oflag = 0;
	newtio.c_lflag = 0;

	newtio.c_cc[VTIME] = 0; // Timeout in deciseconds
	newtio.c_cc[VMIN]  = 0; // Minumum number of characters

	// transmit info to file descriptor
	tcflush(fd, TCIFLUSH);
	tcsetattr(fd, TCSANOW, &newtio); // TCSANOW : change occurs immediately

	sleep(1);
	if (pFile != NULL){

		char strTemp[BUF_MAX];
		char *pStr = NULL;

		while(!feof(pFile)){
			pStr = fgets(strTemp, sizeof(strTemp),pFile);
			if(feof(pFile))
				break;
			printf("%s", strTemp);
			write(fd, strTemp, 30);
			sleep(1);
		}
		fclose(pFile);
	}else{
		fprintf(stderr, "Error\n");
		exit(-1);
	}

	close(fd);
	return 0;

}


