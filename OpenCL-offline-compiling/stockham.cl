__kernel void stockham (           
   const int N,                                 
   const int pN,                                
   const int g,                                 
   const int m,                                 
   const int p,                                 
   const int n,                                 
   __global int *x,                             
   __global int *x_copy,                        
   __global int *rou) {                         
       int j = get_global_id(0);                
       int kk = 0;                              
       int a = 1;                               
       int b = 1 << (n-1);                 
       for (int i = 0; i < n; i++) {            
           int j2 = j << 1;                     
           int j_mod_a = j % a;                 
           int c, d;                            
           if (kk == 0) {                        
               c = x_copy[j];                        
               d = rou[(j_mod_a * b * pN) % (p - 1)] * x_copy[j + (N >> 1)]; 
               d = (d - ((d * m) >> 31) * p);        
               x[j2 - j_mod_a] = c + d > p ? c + d - p : c + d;        
               if (x[j2 - j_mod_a] < 0) x[j2 - j_mod_a] += p; 
               x[j2 - j_mod_a + a] = c < d ? c + p - d : c - d;    
               if (x[j2 - j_mod_a + a] < 0) x[j2 - j_mod_a + a] += p; 
           } else {                                  
               c = x[j];                             
               d = rou[(j_mod_a * b * pN) % (p - 1)] * x[j + (N >> 1)]; 
               d = (d - ((d * m) >> 31) * p);        
               x_copy[j2 - j_mod_a] = (c + d) > p ? c + d - p : c + d;   
               x_copy[j2 - j_mod_a + a] = c < d ? c + p - d : c - d; 
           }                                         
           a <<= 1;                                  
           b >>= 1;                                  
           kk = 1 - kk;                              
       }                                             
}
