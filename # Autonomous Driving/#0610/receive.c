#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>  // c compiler
#include <sys/types.h>
#include <termios.h> // for serial communication
#include <fcntl.h>   // getting or setting properties of files (file control)
#define BUF_MAX 1024

#define FALSE 0
#define TRUE  1

volatile int STOP = FALSE;

int main(void){

	struct termios newtio;
	int fd = 0; // fd : file descriptor
	int res;
	char buf[BUF_MAX];


	// O_RDWR   : Read / Write 
	// O_NOCTTY : limit controls
	fd = open("buffer.txt", O_RDWR | O_NOCTTY);
	if(fd < 0){ // file open fail ...
		fprintf(stderr, "ERROR\n");
		exit(-1); // process end
	}
	
	memset(&newtio, 0, sizeof(newtio)); // Assign dynamic memory

	newtio.c_cflag  = B115200; // Boad Rate
	newtio.c_cflag |= CS8;     // 8bit , parity X , Stopbit 1
	newtio.c_cflag |= CLOCAL;  // Use internal port
	newtio.c_cflag |= CREAD;   // READ ok

	newtio.c_iflag = IGNPAR; // Ignore parity bit (No parity check)
	//newtio.c_iflag = ICRNL; // Carriage Return

	newtio.c_oflag = 0;
	newtio.c_lflag = 0;

	newtio.c_cc[VTIME] = 0; // Timeout in deciseconds
	newtio.c_cc[VMIN]  = 5; // Minumum number of characters (blocking until get 5 characters)


	tcflush(fd, TCIFLUSH);           // tcflush   : Initialize serial port
	tcsetattr(fd, TCSANOW, &newtio); // TCSANOW   : change occurs immediately
					  // tcsetattr : set attributes

	// Get data from file...
	// 1. fgets() : read only 1 line
	// 2. read()  : read datas as much as buffer size
	
	// Loop for Input
	while(STOP==FALSE){	
		res = read(fd, buf, 30);   // res = number of read data 
					    // START,0,L,100,D,1,R,100,D,1,Z -> size 30  
		if (res < 30)
			continue;
		buf[res-1] = 0;            //  \n -> \0 (for printf)
		printf(":%s:%d\n", buf, res); 
		if (buf[0] == 'z') STOP=TRUE;		
		sleep(1);
	}		

	close(fd);
	return 0;

}


