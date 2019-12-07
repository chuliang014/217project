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

__global__ void ComputePhiMag_GPU(float* phiR, float* phiI, float* phiMag, int numK){
    int indexK  = blockIdx.x * K_PHIMAG_BLOCK_SIZE + threadIdx.x;
    if(indexK < numK){
        float real = phiR[indexK];
        float imag = phiI[indexK];
        phiMag[indexK] = real*real + imag*imag;
    }
}

__global__ void computeQ_GPU(int numK, int kGlobalIndex, float* x, float* y, float* z, 
    float* Qr, float* Qi, kValues* kVal){
        int xIndex = blockIdx.x * K_Q_BLOCK_SIZE + threadIdx.x;

        int indexK = 0;
        for(indexK=0; indexK < K_Q_K_ELEMS_PER_GRID && kGlobalIndex < numK; indexK++, kGlobalIndex++){
            float expArg = PIx2 * (kVal[indexK].Kx * x[xIndex] +
                kVal[indexK].Ky * y[xIndex] +
                kVal[indexK].Kz * z[xIndex]);
                
                Qr[xIndex] += kVal[indexK].PhiMag * cosf(expArg);
                Qi[xIndex] += kVal[indexK].PhiMag * sinf(expArg);
        }
}

void ComputePhiMagGPU(int numK, float* d_phiR, float* d_phiI, float* d_phiMag){
    int phiMag_block = (numK-1) / K_PHIMAG_BLOCK_SIZE + 1;
    dim3 DimPhiMagBlock(K_PHIMAG_BLOCK_SIZE,1);
    dim3 DimPhiMagGrid(phiMag_block,1);
    ComputePhiMag_GPU<<<DimPhiMagGrid, DimPhiMagBlock>>>(d_phiR, d_phiI, d_phiMag, numK);
}

void computeQGPU(int numK, int numX,float* d_x, float* d_y, float* d_z,
    kValues* kVal,float* d_Qr, float* d_Qi){
        int gridQ = (numK -1) / K_Q_K_ELEMS_PER_GRID + 1;
        int blockQ = (numX - 1) / K_Q_BLOCK_SIZE + 1;
        dim3 DimQBlock(K_Q_BLOCK_SIZE, 1);
        dim3 DimQGrid(blockQ,1);
        for(int i = 0; i < gridQ; i++){
            int QGridBase = i * K_Q_K_ELEMS_PER_GRID;
            kValues* kValsTile = kVal + QGridBase;
            computeQ_GPU<<<DimQGrid, DimQBlock>>>(numK,QGridBase,d_x,d_y,d_z,d_Qr,d_Qi,kValsTile);
        }
    }

    void createDataStructsCPU(int numK, int numX, float** phiMag,
        float** Qr, float** Qi){
            *phiMag = (float* ) malloc(numK * sizeof(float));
            *Qr = (float*) malloc(numX * sizeof (float));
            *Qi = (float*) malloc(numX * sizeof (float));
        }