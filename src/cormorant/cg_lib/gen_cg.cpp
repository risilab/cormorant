#include <torch/torch.h>
#include <pybind11/stl.h>
#include <vector>
#include <iostream>

// The following code is copyright Imre "Risi" Kondor.
inline double logfact(int n){
  double result=0;
  for(int i=2; i<=n; i++) result+=log((double)i);
  return result;
}

inline double plusminus(int k){ if(k%2==1) return -1; else return +1; }

inline int max( int i, int j ){return (i>j) ? i : j;};
inline int min( int i, int j ){return (i<j) ? i : j;};

double slowCG(int l1, int l2, int l, int m1, int m2, int m){

  int m3=-m;
  int t1=l2-m1-l;
  int t2=l1+m2-l;
  int t3=l1+l2-l;
  int t4=l1-m1;
  int t5=l2+m2;

  int tmin=max(0,max(t1,t2));
  int tmax=min(t3,min(t4,t5));

  double wigner=0;

  double logA=(log(2*l+1)+logfact(l+l1-l2)+logfact(l-l1+l2)+logfact(l1+l2-l)-logfact(l1+l2+l+1))/2;
  logA+=(logfact(l-m3)+logfact(l+m3)+logfact(l1-m1)+logfact(l1+m1)+logfact(l2-m2)+logfact(l2+m2))/2;

  for(int t=tmin; t<=tmax; t++){
    double logB = logfact(t)+logfact(t3-t)+logfact(t4-t)+logfact(t5-t)+logfact(-t1+t)+logfact(-t2+t);
    wigner += plusminus(t)*exp(logA-logB);
    }
  return plusminus(l1-l2-m3)*plusminus(l1-l2+m)*wigner;
}


std::vector<torch::Tensor> GenCGcoeffs(int L) {

    std::vector<torch::Tensor> CGmats;

    std::cout << "Precomputing CG coefficients: ";

    for (int l1 = 0; l1 <= L; l1++){
        std::cout << l1 << " " << std::flush;
        for (int l2 = 0; l2 <= L; l2++){
//            std::cout << "(" << l1 << "," << l2 << ") " << std::flush;
            int lmin = abs(l1-l2);
            int lmax = l1 + l2;
            int N1 = 2*l1+1;
            int N2 = 2*l2+1;
            int N = N1*N2;
            torch::Tensor CGmat = torch::zeros({N1, N2, N}, torch::kDouble);
            for (int l = lmin; l <= lmax; l++){
                int l_off = l*l - lmin*lmin;
//                std::cout << "(" << l1 << "," << l2 << "," << l << ") (" << lmin << "," << lmax << "," << l_off << ") " << CGmat.sizes() << ": "<< std::endl;

                for (int m = -l; m <= l; ++m) {
                    for (int m1 = -l1; m1 <= l1; ++m1) {
                        for (int m2 = -l2; m2 <= l2; ++m2) {
                            if(m == m1+m2) {
//                                std::cout << " (" << m1 << "," << m2 << "," << m << ") "
//                                          << " (" << l1+m1 << "," << l2+m2 << "," << l+m << ") " << l+m+l_off << std::endl;
                                CGmat[l1+m1][l2+m2][l+m+l_off] = slowCG(l1,l2,l,m1,m2,m);
                            } // if
                        } // for m2
                    } // for m1
                } // for m
            } // for l
            CGmats.push_back(CGmat);
        } // for l2
    } // for l1

    std::cout << "Done!" << std::endl;

    return CGmats;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gen_cg_coefffs", &GenCGcoeffs, "Generate CG coefficients.");
}
