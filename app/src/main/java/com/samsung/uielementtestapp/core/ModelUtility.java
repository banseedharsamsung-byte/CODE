package com.samsung.uielementtestapp.core;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

import static java.nio.channels.FileChannel.MapMode.READ_ONLY;

public class ModelUtility {
    private static final String TAG = "ModelUtil";

    public enum INTERPRETER_DELEGATE {
        CPU,
        NPU,
        GPU
    }

    public static Interpreter getInterpreterFromModel(Context context, String model, INTERPRETER_DELEGATE interpreterDelegate) {
        try {
            Interpreter.Options options = new Interpreter.Options();
            options.setUseXNNPACK(false);
            options.setNumThreads(4);

            long frameSaveTime = 0;
            long startWhen = System.currentTimeMillis();
            Log.d(TAG, "LOADING MODEL = " + model);

            AssetFileDescriptor fileDescriptor = context.getAssets().openFd(model);
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long length = fileDescriptor.getDeclaredLength();

            MappedByteBuffer byteBuffer = fileChannel.map(READ_ONLY, startOffset, length);
            Interpreter interpreter = new Interpreter(byteBuffer, options);

            frameSaveTime += System.currentTimeMillis() - startWhen;
            Log.d(TAG, " model Init Time = " + frameSaveTime);

            return interpreter;
        } catch (IOException e) {
            Log.e(TAG, "int interpreter FAILED: " + e.getLocalizedMessage());
            return null;
        }
    }
}

