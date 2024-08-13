__kernel void test (           
   const int N,                                 
   const int pN,                                
   const int a,                                 
   const int b,                                 
   const int g,                                 
   const int m,                                 
   const int p,                                 
   __global int *x,                             
   __global int *x_copy,                        
   __global int *rou,                           
   const int kk) {                              
       int j = get_global_id(0);                
       int j2 = j << 1;                         
       int j_mod_a = j % a;                     
       int c, d;                                
       if (kk == 0) {                            
           c = x_copy[j];                        
           d = rou[(j_mod_a * b * pN) % (p - 1)] * x_copy[j + (N >> 1)]; 
           d = (d - ((d * m) >> 31) * p);        
           if (d >= p) d -= p;                   
           if (d < 0) d += p;                    
           x[j2 - j_mod_a] = (c + d) % p;        
           if (x[j2 - j_mod_a] < 0) x[j2 - j_mod_a] += p; 
           x[j2 - j_mod_a + a] = (c - d) % p;    
           if (x[j2 - j_mod_a + a] < 0) x[j2 - j_mod_a + a] += p; 
       } else {                                  
           c = x[j];                             
           d = rou[(j_mod_a * b * pN) % (p - 1)] * x[j + (N >> 1)]; 
           d = (d - ((d * m) >> 31) * p);        
           if (d >= p) d -= p;                   
           if (d < 0) d += p;                    
           x_copy[j2 - j_mod_a] = (c + d) % p;   
           if (x_copy[j2 - j_mod_a] < 0) x_copy[j2 - j_mod_a] += p; 
           x_copy[j2 - j_mod_a + a] = (c - d) % p; 
           if (x_copy[j2 - j_mod_a + a] < 0) x_copy[j2 - j_mod_a + a] += p; 
       }                                      
}