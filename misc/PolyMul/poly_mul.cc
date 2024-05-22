#include <iostream>
#include <vector>

int main(){
	int n = 4;
	std::vector<int> a(n);
	std::vector<int> b(n);
	std::vector<int> out(n);

	for (int i = 0; i < n; i++){
		a[i] = i;
		b[i] = i;
		out[i] = 0;
	}
	
	/*
	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; j++){
			out[i + j] += a[i] * b[j];
		}
	}
	for (int i = 0; i < n + 1; i++){
		std::cout << "out[" << i << "]= " << out[i] << std::endl;
	}

	for (int i = 0; i < n + 1; i++){
		out[i] = 0;
	}
	*/
	
	for (int i = 0; i < n; i++){
		int r = 0;
		for (int j = 0; j <= i; j++){
			//std::cout << "(+) i= " << i << " j= " << j << " a[j]= " << a[j] << " b[i-j]= " << b[i - j] << std::endl;
			std::cout << "(" << j << ", " << i - j << ")" << std::endl;
			r += a[j] * b[i - j];
		}
		for (int j = i + 1; j < n; j++){
			//std::cout << "(-) i= " << i << " j= " << j << " a[j]= " << a[j] << " b[n-j+i]= " << b[n - j + i] << std::endl;
			std::cout << "(" << j << ", " << n - j + i << ")" << std::endl;
			r -= a[j] * b[n - j + i];
		}
		out[i] = r;
	}
	for (int i = 0; i < n; i++){
		std::cout << "out[" << i << "]= " << out[i] << std::endl;
	}
}
