/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.neuralgis;

import static jcuda.driver.CUdevice_attribute.*;
import static jcuda.driver.JCudaDriver.*;

import com.neuralgis.window.MainFrame;
import javax.swing.UIManager;
import jcuda.Pointer;
import jcuda.runtime.JCuda;
import com.jtattoo.plaf.noire.NoireLookAndFeel;
import com.neuralgis.graphics.GPUMathKernel;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import jcuda.driver.*;
/**
 *
 * @author braunth
 */
public class Exec {
    
    public static MainFrame frame;
    
    public static void main(String... args){
        try
{
  UIManager.setLookAndFeel("com.jtattoo.plaf.acryl.AcrylLookAndFeel");
  //Another way is to use the #setLookAndFeel method of the SyntheticaLookAndFeel class
  //SyntheticaLookAndFeel.setLookAndFeel(String className, boolean antiAlias, boolean useScreenMenuOnMac);
}
catch (Exception e)
{
  e.printStackTrace();
}

        loadf();
        JCuda.cudaSetDevice(0);
        Pointer pointer = new Pointer();
        JCuda.cudaMalloc(pointer, 4);
        System.out.println("Pointer: "+pointer);
        JCuda.cudaFree(pointer);
        GPUMathKernel kernel = new GPUMathKernel();
        System.out.println("Addition of 2000 and 4000: " + kernel.mult(2000, 4000) + "\n");
        profile(250000000, kernel);
        frame = new MainFrame();
        frame.setVisible(true);
    }
    
    private static void profile(int len, GPUMathKernel kern){
        System.out.println("Profiling GPU float array multiplication for " + len + " items (" + (len*4/1000000) + "Mb)");
        long millis0 = System.currentTimeMillis();
        float[] one = fillArray(len);
        float[] two = fillArray(len);
        System.out.println("Done loading random arrays in " + (System.currentTimeMillis() - millis0) +"ms, now testing GPU...\nTesting add");
        millis0 = System.currentTimeMillis();
        float[] r0 = kern.pmult(2,one, two);
        System.out.println("Done testing add in " + (System.currentTimeMillis() - millis0) +"ms, now testing sub| " + one[0] + "+" + two[0] + "=" + r0[0]);
        millis0 = System.currentTimeMillis();
        /*float[] r2 = kern.sub(one, two);
        System.out.println("Done testing sub in " + (System.currentTimeMillis() - millis0) +"ms, now testing mult");
        millis0 = System.currentTimeMillis();
        float[] r3 = kern.mult(one, two);
        System.out.println("Done testing mult in " + (System.currentTimeMillis() - millis0) +"ms, now testing div");
        millis0 = System.currentTimeMillis();
        float[] r4 = kern.div(one, two);
        System.out.println("Done testing div in " + (System.currentTimeMillis() - millis0) + ", freeing objects...");
        //r0= null;
        r2= null;
        r3= null;
        r4= null;*/
        millis0 = System.currentTimeMillis();
        for (int i = 0; i < len; i++){
            r0[i] = one[i] + two[i];
        }
        System.out.println("Done testing cpu-add in " + (System.currentTimeMillis() - millis0));
        
    }
    
    
    private static float[] fillArray(int len){
        Random rand = new Random();
        float[] ret = new float[len];
        
        for (int i = 0; i  < len; i++) ret[i] = rand.nextFloat();
        
        return ret;
    }
 
    private static void loadf(){
         JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);

        // Obtain the number of devices
        int deviceCountArray[] = { 0 };
        cuDeviceGetCount(deviceCountArray);
        int deviceCount = deviceCountArray[0];
        System.out.println("Found " + deviceCount + " devices");

        for (int i = 0; i < deviceCount; i++)
        {
            CUdevice device = new CUdevice();
            cuDeviceGet(device, i);

            // Obtain the device name
            byte deviceName[] = new byte[1024];
            cuDeviceGetName(
                deviceName, deviceName.length, device);
            String name = createString(deviceName);

            // Obtain the compute capability
            int majorArray[] = { 0 };
            int minorArray[] = { 0 };
            cuDeviceComputeCapability(
                majorArray, minorArray, device);
            int major = majorArray[0];
            int minor = minorArray[0];

            System.out.println("Device " + i + ": " + name + 
                " with Compute Capability " + major + "." + minor);
            deviceData = deviceData + "Device " + i + ": " + name + " with Compute Capability " + major + "." + minor + "\n";
            
            // Obtain the device attributes
            int array[] = { 0 };
            List<Integer> attributes = getAttributes();
            for (Integer attribute : attributes)
            {
                String description = getAttributeDescription(attribute);
                cuDeviceGetAttribute(array, attribute, device);
                int value = array[0];
                
                System.out.printf("    %-52s : %d\n", description, value);
                deviceData += description + ": " + value + "\n";
                       
            }
        }
    }
    
    public static String deviceData = "";
    
    private static String getAttributeDescription(int attribute)
    {
        switch (attribute)
        {
            case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: 
                return "Maximum number of threads per block";
            case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: 
                return "Maximum x-dimension of a block";
            case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: 
                return "Maximum y-dimension of a block";
            case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: 
                return "Maximum z-dimension of a block";
            case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: 
                return "Maximum x-dimension of a grid";
            case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: 
                return "Maximum y-dimension of a grid";
            case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: 
                return "Maximum z-dimension of a grid";
            case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: 
                return "Maximum shared memory per thread block in bytes";
            case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY: 
                return "Total constant memory on the device in bytes";
            case CU_DEVICE_ATTRIBUTE_WARP_SIZE: 
                return "Warp size in threads";
            case CU_DEVICE_ATTRIBUTE_MAX_PITCH: 
                return "Maximum pitch in bytes allowed for memory copies";
            case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK: 
                return "Maximum number of 32-bit registers per thread block";
            case CU_DEVICE_ATTRIBUTE_CLOCK_RATE: 
                return "Clock frequency in kilohertz";
            case CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT: 
                return "Alignment requirement";
            case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: 
                return "Number of multiprocessors on the device";
            case CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT: 
                return "Whether there is a run time limit on kernels";
            case CU_DEVICE_ATTRIBUTE_INTEGRATED: 
                return "Device is integrated with host memory";
            case CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY: 
                return "Device can map host memory into CUDA address space";
            case CU_DEVICE_ATTRIBUTE_COMPUTE_MODE: 
                return "Compute mode";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH: 
                return "Maximum 1D texture width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH: 
                return "Maximum 2D texture width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT: 
                return "Maximum 2D texture height";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH: 
                return "Maximum 3D texture width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT: 
                return "Maximum 3D texture height";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH: 
                return "Maximum 3D texture depth";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH: 
                return "Maximum 2D layered texture width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT: 
                return "Maximum 2D layered texture height";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS: 
                return "Maximum layers in a 2D layered texture";
            case CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT: 
                return "Alignment requirement for surfaces";
            case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS: 
                return "Device can execute multiple kernels concurrently";
            case CU_DEVICE_ATTRIBUTE_ECC_ENABLED: 
                return "Device has ECC support enabled";
            case CU_DEVICE_ATTRIBUTE_PCI_BUS_ID: 
                return "PCI bus ID of the device";
            case CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID: 
                return "PCI device ID of the device";
            case CU_DEVICE_ATTRIBUTE_TCC_DRIVER: 
                return "Device is using TCC driver model";
            case CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: 
                return "Peak memory clock frequency in kilohertz";
            case CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: 
                return "Global memory bus width in bits";
            case CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: 
                return "Size of L2 cache in bytes";
            case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: 
                return "Maximum resident threads per multiprocessor";
            case CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT: 
                return "Number of asynchronous engines";
            case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING: 
                return "Device shares a unified address space with the host";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH: 
                return "Maximum 1D layered texture width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS: 
                return "Maximum layers in a 1D layered texture";
            case CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID: 
                return "PCI domain ID of the device";
        }
        return "(UNKNOWN ATTRIBUTE)";
    }
    
    /**
     * Returns a list of all CUdevice_attribute constants
     * 
     * @return A list of all CUdevice_attribute constants
     */
    private static List<Integer> getAttributes()
    {
        List<Integer> list = new ArrayList<Integer>();
        list.add(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
        list.add(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY);
        list.add(CU_DEVICE_ATTRIBUTE_WARP_SIZE);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_PITCH);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK);
        list.add(CU_DEVICE_ATTRIBUTE_CLOCK_RATE);
        list.add(CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT);
        list.add(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
        list.add(CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT);
        list.add(CU_DEVICE_ATTRIBUTE_INTEGRATED);
        list.add(CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY);
        list.add(CU_DEVICE_ATTRIBUTE_COMPUTE_MODE);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS);
        list.add(CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT);
        list.add(CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS);
        list.add(CU_DEVICE_ATTRIBUTE_ECC_ENABLED);
        list.add(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID);
        list.add(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID);
        list.add(CU_DEVICE_ATTRIBUTE_TCC_DRIVER);
        list.add(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE);
        list.add(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);
        list.add(CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT);
        list.add(CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS);
        list.add(CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID);
        return list;
    }

    /**
     * Creates a String from a zero-terminated string in a byte array
     * 
     * @param bytes
     *            The byte array
     * @return The String
     */
    private static String createString(byte bytes[])
    {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < bytes.length; i++)
        {
            char c = (char)bytes[i];
            if (c == 0)
            {
                break;
            }
            sb.append(c);
        }
        return sb.toString();
    }
    
}

