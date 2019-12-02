#include <cstdlib>

#define PI   3.1415926535897932384626433832795029f
#define PIx2 6.2831853071795864769252867665590058f

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define K_ELEMS_PER_GRID  2048
#define K_PHIMAG_BLOCK_SIZE 512
#define K_Q_BLOCK_SIZE 256
#define K_Q_K_ELEMS_PER_GRID 1024

struct kValues {
    float Kx;
    float Ky;
    float Kz;
    float PhiMag;
  };

  __constant__ __device__ kValues kVal[K_Q_K_ELEMS_PER_GRID];

__global__ void ComputePhiMag_GPU(float* phiR, float* phiI, float* phiMag, int numK){
    int indexK  = blockIdx.x * K_PHIMAG_BLOCK_SIZE + threadIdx.x;
    if(indexK < numK){
        float real = phiR[indexK];
        float imag = phiI[indexK];
        phiMag[indexK] = real*real + imag*imag;
    }
}

__global__ void computeQ_GPU(int numK, int kGlobalIndex, float* x, float* y, float* z, 
    float* Qr, float* Qi){
        __shared__ float s_x,s_y,s_z,s_Qr,s_Qi;

        int xIndex = blockIdx.x * K_Q_BLOCK_SIZE + threadIdx.x;
        
        

        s_x = x[xIndex];
        s_y = y[xIndex];
        s_z = z[xIndex];
        s_Qr = Qr[xIndex];
        s_Qi = Qi[xIndex];
        int indexK = 0;
        for(indexK=0; indexK < K_Q_K_ELEMS_PER_GRID && kGlobalIndex < numK; indexK++, kGlobalIndex++){
            float expArg = PIx2 * (kVal[indexK].Kx * s_x +
                kVal[indexK].Ky * s_y +
                kVal[indexK].Kz * s_z);
                
                s_Qr += kVal[indexK].PhiMag * cosf(expArg);
                s_Qi += kVal[indexK].PhiMag * sinf(expArg);
        }
        Qr[xIndex] = s_Qr;
        Qi[xIndex] = s_Qi;
}

void ComputePhiMagGPU(int numK, float* d_phiR, float* d_phiI, float* d_phiMag){
    int phiMag_block = (numK-1) / K_PHIMAG_BLOCK_SIZE + 1;
    dim3 DimPhiMagBlock(K_PHIMAG_BLOCK_SIZE,1);
    dim3 DimPhiMagGrid(phiMag_block,1);
    ComputePhiMag_GPU<<<DimPhiMagGrid, DimPhiMagBlock>>>(d_phiR, d_phiI, d_phiMag, numK);
}

void computeQGPU(int numK, int numX,float* d_x, float* d_y, float* d_z,
    kValues* kVals,float* d_Qr, float* d_Qi){
        int gridQ = (numK -1) / K_Q_K_ELEMS_PER_GRID + 1;
        int blockQ = (numX - 1) / K_Q_BLOCK_SIZE + 1;
        dim3 DimQBlock(K_Q_BLOCK_SIZE, 1);
        dim3 DimQGrid(blockQ,1);
        for(int i = 0; i < gridQ; i++){
            int QGridBase = i * K_Q_K_ELEMS_PER_GRID;
            kValues* kValsTile = kVals + QGridBase;
            int num = MIN(K_Q_K_ELEMS_PER_GRID, numK - QGridBase);
            cudaMemcpyToSymbol(kVal, kValsTile, num * sizeof(kValues), 0);
            computeQ_GPU<<<DimQGrid, DimQBlock>>>(numK,QGridBase,d_x,d_y,d_z,d_Qr,d_Qi);
        }
    }

    void createDataStructsCPU(int numK, int numX, float** phiMag,
        float** Qr, float** Qi){
            *phiMag = (float* ) malloc(numK * sizeof(float));
            *Qr = (float*) malloc(numX * sizeof (float));
            *Qi = (float*) malloc(numX * sizeof (float));
        }