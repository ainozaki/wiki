#include <stdio.h>

#define ARRAYSZ 10

int main(){
	/*
		array1[0] = 0
	 &array1[0] = 0x16d7d3658
		array1    = 0x16d7d3658
	*/
	char array1[ARRAYSZ] = {0};
	printf(" array1[0] = %d\n", array1[0]);
	printf("&array1[0] = %p\n", &array1[0]);
	printf(" array1    = %p\n", array1);

	/*
		  array2[0] = array2
		 &array2[0] = 0x16b04b608
 			array2    = 0x16b04b608
	*/
	char *array2[ARRAYSZ];
	array2[0] = "array2";
	printf(" array2[0] = %s\n", array2[0]);
	printf("&array2[0] = %p\n", &array2[0]);
	printf(" array2    = %p\n", array2);
}
