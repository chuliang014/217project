#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include "malloc.h"
#include "string.h" 
#include "parboil.h"

#include "file.h"
#include "computeQ.cu"

int main(int argc, char *argv[]){
    int numX, numK;		/* Number of X and K values */
  int original_numK;		/* Number of K values in input file */
  float *kx, *ky, *kz;		/* K trajectory (3D vectors) */
  float *x, *y, *z;		/* X coordinates (3D vectors) */
  float *phiR, *phiI;		/* Phi values (complex) */
  float *phiMag;		/* Magnitude of Phi */
  float *Qr, *Qi;		/* Q signal (complex) */
  struct kValues* kVals;

  struct pb_Parameters *params;
  struct pb_TimerSet timers;

  pb_InitializeTimerSet(&timers);

  /* Read command line */
  params = pb_ReadParameters(&argc, argv);
  if ((params->inpFiles[0] == NULL) || (params->inpFiles[1] != NULL))
    {

      fprintf(stderr, "Expecting one input filename\n");
      exit(-1);
    }

    /* Read in data */
  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  inputData(params->inpFiles[0],
	    &original_numK, &numX,
	    &kx, &ky, &kz,
	    &x, &y, &z,
        &phiR, &phiI);
        
    /* Reduce the number of k-space samples if a number is given
   * on the command line */
  if (argc < 2)
  numK = original_numK;
else
  {
    int inputK;
    char *end;
    inputK = strtol(argv[1], &end, 10);
    if (end == argv[1])
  {
    fprintf(stderr, "Expecting an integer parameter\n");
    exit(-1);
  }

    numK = MIN(inputK, original_numK);
  }

  printf("%d pixels in output; %d samples in trajectory; using %d samples\n",
         numX, original_numK, numK);

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  /* Create CPU data structures */
  createDataStructsCPU(numK, numX, &phiMag, &Qr, &Qi);

   //vincent compute phimag in GPU
   {
       float *d_phiI, *d_phiR, *d_phiMag;
       pb_SwitchToTimer(&timers, pb_TimerID_COPY);

       //vincent allocate memory and cp data to gpu
       cudaMalloc ((void **) &d_phiR, sizeof(float) * numK);
       cudaMemcpy (d_phiR, phiR, sizeof(float) * numK, cudaMemcpyHostToDevice);

       cudaMalloc ((void **) &d_phiI, sizeof(float) * numK);
       cudaMemcpy (d_phiI, phiI, sizeof(float) * numK, cudaMemcpyHostToDevice);

       cudaMalloc((void **)&d_phiMag, sizeof(float) * numK);
       cudaThreadSynchronize();

       pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
       
       ComputePhiMagGPU(numK, d_phiR,d_phiI,d_phiMag);
       cudaThreadSynchronize();

       pb_SwitchToTimer(&timers, pb_TimerID_COPY);

       //int num, int size, float *& dev_ptr, float * host_ptr
       //vincent cp data to cpu from gpu
       cudaMemcpy (phiMag, d_phiMag, numK * sizeof(float), cudaMemcpyDeviceToHost);
       cudaFree(d_phiMag);
       cudaFree(d_phiR);
       cudaFree(d_phiI);
   }

   pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

   kVals = (struct kValues*)calloc(numK, sizeof (struct kValues));
  int k;
  for (k = 0; k < numK; k++) {
    kVals[k].Kx = kx[k];
    kVals[k].Ky = ky[k];
    kVals[k].Kz = kz[k];
    kVals[k].PhiMag = phiMag[k];
  }

  free(phiMag);

  //vincent computeQ in GPU
  {
    float *d_x, *d_y, *d_z;
    float *d_Qr, *d_Qi;
    kValues* d_kVal;
    //vincent allocate memory and cp data to GPU
    pb_SwitchToTimer(&timers, pb_TimerID_COPY);

    cudaMalloc ((void **) &d_x, sizeof(float) * numX);
    cudaMemcpy (d_x, x, sizeof(float) * numX, cudaMemcpyHostToDevice);

    cudaMalloc ((void **) &d_y, sizeof(float) * numX);
    cudaMemcpy (d_y, y, sizeof(float) * numX, cudaMemcpyHostToDevice);

    cudaMalloc ((void **) &d_z, sizeof(float) * numX);
    cudaMemcpy (d_z, z, sizeof(float) * numX, cudaMemcpyHostToDevice);

    cudaMalloc ((void **) &d_kVal, sizeof(struct kValues) * numK);
    cudaMemcpy (d_kVal, kVals, sizeof(struct kValues) * numK, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_Qr, numX * sizeof(float));
    cudaMemset((void *)d_Qr, 0, numX * sizeof(float));

    cudaMalloc((void **)&d_Qi, numX * sizeof(float));
    cudaMemset((void *)d_Qi, 0, numX * sizeof(float));
    cudaThreadSynchronize();
    //vincent start compute 
    pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
    computeQGPU(numK, numX, d_x, d_y, d_z, d_kVal, d_Qr, d_Qi);
    cudaThreadSynchronize();

    pb_SwitchToTimer(&timers, pb_TimerID_COPY);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_kVal);

    //vincent cp data from gpu to cpu
    cudaMemcpy (d_Qr, Qr, numX * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_Qr);

    cudaMemcpy (d_Qi, Qi, numX * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_Qi);
  }
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

    if (params->outFile)
    {
      /* Write Q to file */
      pb_SwitchToTimer(&timers, pb_TimerID_IO);
      outputData(params->outFile, Qr, Qi, numX);
      pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
    }

    free (kx);
  free (ky);
  free (kz);
  free (x);
  free (y);
  free (z);
  free (phiR);
  free (phiI);
  free (phiMag);
  free (kVals);
  free (Qr);
  free (Qi);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(params);

  return 0;
}