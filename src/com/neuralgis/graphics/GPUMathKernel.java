package com.neuralgis.graphics;

import java.util.Arrays;
import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.jcublas.JCublas.*;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas;
import jcuda.vec.VecFloat;

/**
 *
 * @author braunth
 */
public class GPUMathKernel {

    public GPUMathKernel() {

        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);
        VecFloat.init();
        JCublas.cublasInit();
    }

    public void init() {
        VecFloat.init();
        JCublas.cublasInit();
    }

    public void shutdown() {
        VecFloat.shutdown();
        JCublas.cublasShutdown();
    }

    public float[] padd(float[] raw1, float[] raw2) {
        if ((raw1.length != raw2.length) || (raw1.length + raw2.length < 2)) {
            throw new IndexOutOfBoundsException("Incompatible float arrays");
        }
        Pointer p1 = new Pointer();
        Pointer p2 = new Pointer();
        //a2 = new float[a1.length];
        float n2old = raw2[0];
        System.out.println(raw1[0] + "+" + raw2[0]);
        int length = raw1.length;
        cublasAlloc(length, Sizeof.FLOAT, p1);
        cublasAlloc(length, Sizeof.FLOAT, p2);

        cublasSetVector(length, Sizeof.FLOAT, Pointer.to(raw1), 1, p1, 1);
        cublasSetVector(length, Sizeof.FLOAT, Pointer.to(raw2), 1, p2, 1);

        //Saxpy: ax + y
        // THE ACTUAL OPERATION: MULTIPLY AND ADD
        JCublas.cublasSaxpy(length, 1f, p1, 1, p2, 1);

        cublasGetVector(length, Sizeof.FLOAT, p2, 1, Pointer.to(raw2), 1);
        System.out.println(n2old + "->" + raw2[0]);

        cublasFree(p1);
        cublasFree(p2);
        return raw2;
    }

    public float[] psub(float[] raw1, float[] raw2) {
        if ((raw1.length != raw2.length) || (raw1.length + raw2.length < 2)) {
            throw new IndexOutOfBoundsException("Incompatible float arrays");
        }
        Pointer p1 = new Pointer();
        Pointer p2 = new Pointer();
        //a2 = new float[a1.length];
        float n2old = raw2[0];
        System.out.println(raw1[0] + "-" + raw2[0]);
        int length = raw1.length;
        cublasAlloc(length, Sizeof.FLOAT, p1);
        cublasAlloc(length, Sizeof.FLOAT, p2);

        cublasSetVector(length, Sizeof.FLOAT, Pointer.to(raw1), 1, p1, 1);
        cublasSetVector(length, Sizeof.FLOAT, Pointer.to(raw2), 1, p2, 1);

        //Saxpy: ax + y
        // THE ACTUAL OPERATION: MULTIPLY AND ADD
        JCublas.cublasSaxpy(length, -1f, p2, 1, p1, 1);

        cublasGetVector(length, Sizeof.FLOAT, p1, 1, Pointer.to(raw1), 1);
        System.out.println(n2old + "->" + raw1[0]);

        cublasFree(p1);
        cublasFree(p2);
        return raw1;
    }
    
    public float[] pmult(float scalar, float[] raw1, float[] raw2)
    {
        if ((raw1.length != raw2.length) || (raw1.length + raw2.length < 2)) {
            throw new IndexOutOfBoundsException("Incompatible float arrays");
        }
        int n = raw1.length;
        float[] output = new float[n];
        Arrays.fill(output, 1);
        float beta = 0;
        System.out.println(raw1[0] + "*" + raw2[0]);

        // Allocate memory on the device
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();
        cublasAlloc(n, Sizeof.FLOAT, d_A);
        cublasAlloc(n, Sizeof.FLOAT, d_B);
        cublasAlloc(n, Sizeof.FLOAT, d_C);

        // Copy the memory from the host to the device
        cublasSetVector(n, Sizeof.FLOAT, Pointer.to(raw1), 1, d_A, 1);
        cublasSetVector(n, Sizeof.FLOAT, Pointer.to(raw2), 1, d_B, 1);
        cublasSetVector(n, Sizeof.FLOAT, Pointer.to(output), 1, d_C, 1);

        // Execute sgemm
        cublasSgemm('n', 'n', n, n, n, scalar, d_A, n, d_B, n, beta, d_C, n);

        // Copy the result from the device to the host
        cublasGetVector(n, Sizeof.FLOAT, d_C, 1, Pointer.to(output), 1);
        System.out.println("Result: " + output[0]);
        // Clean up
        cublasFree(d_A);
        cublasFree(d_B);
        cublasFree(d_C);
        return output;
    }

    public float add(float one, float two) {
        return this.add(new float[]{one}, new float[]{two})[0];
    }

    public float[] add(float[] raw1, float[] raw2) {

        if ((raw1.length != raw2.length) || (raw1.length + raw2.length < 2)) {
            throw new IndexOutOfBoundsException("Incompatible float arrays");
        }

        int n = raw1.length;
        CUdeviceptr deviceX = new CUdeviceptr();
        cuMemAlloc(deviceX, n * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceX, Pointer.to(raw1), n * Sizeof.FLOAT);

        CUdeviceptr deviceY = new CUdeviceptr();
        cuMemAlloc(deviceY, n * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceY, Pointer.to(raw2), n * Sizeof.FLOAT);

        CUdeviceptr deviceResult = new CUdeviceptr();
        cuMemAlloc(deviceResult, n * Sizeof.FLOAT);

        VecFloat.add(n, deviceResult, deviceX, deviceY); // result = x+y

        // Allocate host output memory and copy the device output
        // to the host.
        float hostResult[] = new float[n];
        cuMemcpyDtoH(Pointer.to(hostResult), deviceResult, n * Sizeof.FLOAT);

        cuMemFree(deviceX);
        cuMemFree(deviceY);
        cuMemFree(deviceResult);

        return hostResult;
    }

    public float sub(float one, float two) {
        return this.sub(new float[]{one}, new float[]{two})[0];
    }

    public float[] sub(float[] raw1, float[] raw2) {

        if ((raw1.length != raw2.length) || (raw1.length + raw2.length < 2)) {
            throw new IndexOutOfBoundsException("Incompatible float arrays");
        }

        int n = raw1.length;
        CUdeviceptr deviceX = new CUdeviceptr();
        cuMemAlloc(deviceX, n * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceX, Pointer.to(raw1), n * Sizeof.FLOAT);

        CUdeviceptr deviceY = new CUdeviceptr();
        cuMemAlloc(deviceY, n * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceY, Pointer.to(raw2), n * Sizeof.FLOAT);

        CUdeviceptr deviceResult = new CUdeviceptr();
        cuMemAlloc(deviceResult, n * Sizeof.FLOAT);

        VecFloat.sub(n, deviceResult, deviceX, deviceY); // result = x+y

        // Allocate host output memory and copy the device output
        // to the host.
        float hostResult[] = new float[n];
        cuMemcpyDtoH(Pointer.to(hostResult), deviceResult, n * Sizeof.FLOAT);

        cuMemFree(deviceX);
        cuMemFree(deviceY);
        cuMemFree(deviceResult);

        return hostResult;
    }

    public float mult(float one, float two) {
        return this.mult(new float[]{one}, new float[]{two})[0];
    }

    public float[] mult(float[] raw1, float[] raw2) {

        if ((raw1.length != raw2.length) || (raw1.length + raw2.length < 2)) {
            throw new IndexOutOfBoundsException("Incompatible float arrays");
        }

        int n = raw1.length;
        CUdeviceptr deviceX = new CUdeviceptr();
        cuMemAlloc(deviceX, n * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceX, Pointer.to(raw1), n * Sizeof.FLOAT);

        CUdeviceptr deviceY = new CUdeviceptr();
        cuMemAlloc(deviceY, n * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceY, Pointer.to(raw2), n * Sizeof.FLOAT);

        CUdeviceptr deviceResult = new CUdeviceptr();
        cuMemAlloc(deviceResult, n * Sizeof.FLOAT);

        VecFloat.mul(n, deviceResult, deviceX, deviceY); // result = x+y

        // Allocate host output memory and copy the device output
        // to the host.
        float hostResult[] = new float[n];
        cuMemcpyDtoH(Pointer.to(hostResult), deviceResult, n * Sizeof.FLOAT);

        cuMemFree(deviceX);
        cuMemFree(deviceY);
        cuMemFree(deviceResult);

        return hostResult;
    }

    public float div(float one, float two) {
        return this.div(new float[]{one}, new float[]{two})[0];
    }

    public float[] div(float[] raw1, float[] raw2) {

        if ((raw1.length != raw2.length) || (raw1.length + raw2.length < 2)) {
            throw new IndexOutOfBoundsException("Incompatible float arrays");
        }

        int n = raw1.length;
        CUdeviceptr deviceX = new CUdeviceptr();
        cuMemAlloc(deviceX, n * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceX, Pointer.to(raw1), n * Sizeof.FLOAT);

        CUdeviceptr deviceY = new CUdeviceptr();
        cuMemAlloc(deviceY, n * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceY, Pointer.to(raw2), n * Sizeof.FLOAT);

        CUdeviceptr deviceResult = new CUdeviceptr();
        cuMemAlloc(deviceResult, n * Sizeof.FLOAT);

        VecFloat.div(n, deviceResult, deviceX, deviceY); // result = x+y

        // Allocate host output memory and copy the device output
        // to the host.
        float hostResult[] = new float[n];
        cuMemcpyDtoH(Pointer.to(hostResult), deviceResult, n * Sizeof.FLOAT);

        cuMemFree(deviceX);
        cuMemFree(deviceY);
        cuMemFree(deviceResult);
        return hostResult;
    }

    public void doTest() {
        VecFloat.init();

        // Allocate and fill the host input data
        int n = 50000;
        float hostX[] = new float[n];
        float hostY[] = new float[n];
        for (int i = 0; i < n; i++) {
            hostX[i] = (float) i;
            hostY[i] = (float) i;
        }

        // Allocate the device pointers, and copy the
        // host input data to the device
        CUdeviceptr deviceX = new CUdeviceptr();
        cuMemAlloc(deviceX, n * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceX, Pointer.to(hostX), n * Sizeof.FLOAT);

        CUdeviceptr deviceY = new CUdeviceptr();
        cuMemAlloc(deviceY, n * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceY, Pointer.to(hostY), n * Sizeof.FLOAT);

        CUdeviceptr deviceResult = new CUdeviceptr();
        cuMemAlloc(deviceResult, n * Sizeof.FLOAT);

        // Perform the vector operations
        VecFloat.cos(n, deviceX, deviceX);               // x = cos(x)  
        VecFloat.mul(n, deviceX, deviceX, deviceX);      // x = x*x
        VecFloat.sin(n, deviceY, deviceY);               // y = sin(y)
        VecFloat.mul(n, deviceY, deviceY, deviceY);      // y = y*y
        VecFloat.add(n, deviceResult, deviceX, deviceY); // result = x+y

        // Allocate host output memory and copy the device output
        // to the host.
        float hostResult[] = new float[n];
        cuMemcpyDtoH(Pointer.to(hostResult), deviceResult, n * Sizeof.FLOAT);

        // Verify the result
        boolean passed = true;
        for (int i = 0; i < n; i++) {
            float expected = (float) (Math.cos(hostX[i]) * Math.cos(hostX[i])
                    + Math.sin(hostY[i]) * Math.sin(hostY[i]));
            if (Math.abs(hostResult[i] - expected) > 1e-5) {
                System.out.println(
                        "At index " + i + " found " + hostResult[i]
                        + " but expected " + expected);
                passed = false;
                break;
            }
        }
        System.out.println("Test " + (passed ? "PASSED" : "FAILED"));

        // Clean up.
        cuMemFree(deviceX);
        cuMemFree(deviceY);
        cuMemFree(deviceResult);

    }

}
