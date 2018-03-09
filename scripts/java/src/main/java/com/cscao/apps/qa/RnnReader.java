package com.cscao.apps.qa;

import com.alibaba.fastjson.JSONReader;
import com.linkedin.paldb.api.Configuration;
import com.linkedin.paldb.api.PalDB;
import com.linkedin.paldb.api.StoreReader;
import org.jetbrains.bio.npy.NpzFile;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Map;
import java.util.StringTokenizer;
import java.util.stream.Collectors;

public class RnnReader {
    private StoreReader mEmbReader;

    public RnnReader() {
        Configuration config = PalDB.newConfiguration();
        config.set(Configuration.CACHE_ENABLED, "true");
        config.set(Configuration.MMAP_SEGMENT_SIZE, "104857600"); // 1024^2*100 = 100 MB
        config.set(Configuration.CACHE_BYTES, "104857600");
        mEmbReader = PalDB.createReader(new File(GenEmbDB.EMB_DB_PATH), config);
    }

    private static final int VECTOR_DIM = 300;

    public String getPath(String filename) {
        return getClass().getClassLoader().getResource(filename).getPath();
    }

    private Feature getFeature(String filename) throws FileNotFoundException {
        JSONReader jsonReader = new JSONReader(new FileReader(getPath(filename)));
        return jsonReader.readObject(Feature.class);
    }

    private float[] getEmbeddings(Feature feature) throws FileNotFoundException {
        String[] words = feature.getWords();
        float[] feed = new float[words.length * VECTOR_DIM];
        for (int j = 0; j < words.length; j++) {
            float[] vector = mEmbReader.get(words[j]);
            if (vector != null) {
                System.arraycopy(vector, 0, feed, j * VECTOR_DIM, VECTOR_DIM);
            }
        }

        return feed;
    }

    private void benchmark() throws IOException {
        int numFeatures = 3;
        int batch = 2;
        String[] docIDs = new String[]{"39433103_6.json", "265489_9.json"};
        long start = System.currentTimeMillis();
        Feature qFeature = getFeature("how-tall-is-mount-mckinley.json");
        String[] qWords = qFeature.getWords();

        float[] x2Feed = getEmbeddings(qFeature);
        float[] x2FeedBatch = new float[batch * x2Feed.length];
        for (int b = 0; b < batch; b++) {
            System.arraycopy(x2Feed, 0, x2FeedBatch,
                    b * x2Feed.length, x2Feed.length);
        }
        byte[] x2mFeedBatch = new byte[batch * qWords.length];

        int[] docLengths = new int[batch];
        Feature[] batchFeatures = new Feature[batch];
        for (int b = 0; b < batch; b++) {
            Feature docFeature = getFeature(docIDs[b]);
            batchFeatures[b] = docFeature;
            docLengths[b] = docFeature.getWords().length;
        }

        int maxDocLength = Arrays.stream(docLengths).summaryStatistics().getMax();

        byte[] x1mFeedBatch = new byte[batch * maxDocLength];
        float[] x1FeedBatch = new float[batch * maxDocLength * VECTOR_DIM];
        float[] x1fFeedBatch = new float[batch * maxDocLength * numFeatures];

        for (int b = 0; b < batch; b++) {
            for (int p = docLengths[b]; p < maxDocLength; p++) {
                x1mFeedBatch[b * maxDocLength + p] = 1;
            }

            Feature docFeature = batchFeatures[b];

            float[] x1Feed = getEmbeddings(docFeature);
            System.arraycopy(x1Feed, 0, x1FeedBatch,
                    b * maxDocLength * VECTOR_DIM, x1Feed.length);

            String[] docWords = docFeature.getWords();
//                    float[] docTf = docFeature.getTf();
            Map<String, Integer> wordCounter = Arrays.stream(docWords)
                    .collect(Collectors.toMap(String::toLowerCase, w -> 1, Integer::sum));
            float[] x1fFeed = new float[docWords.length * numFeatures];
            for (int k = 0; k < docWords.length; k++) {
                String docWord = docWords[k];
                for (String qWord : qWords) {
                    if (docWord.equals(qWord)) {
                        x1fFeed[k * 3] = 1.0f;
                    }

                    if (docWord.equalsIgnoreCase(qWord)) {
                        x1fFeed[k * 3 + 1] = 1.0f;
                    }
                }
//                        x1fFeed[k * 3 + 2] = docTf[k];
                x1fFeed[k * 3 + 2] = wordCounter.get(docWord.toLowerCase())
                        * 1.0f / docWords.length;
            }

            System.arraycopy(x1fFeed, 0, x1fFeedBatch,
                    b * maxDocLength * numFeatures, x1fFeed.length);

        }

        float[][] inputs = new float[3][];
        byte[][] inputMasks = new byte[2][];

        inputs[0] = x1FeedBatch;
        inputs[1] = x1fFeedBatch;
        inputs[2] = x2FeedBatch;

        inputMasks[0] = x1mFeedBatch;
        inputMasks[1] = x2mFeedBatch;


//        float[][] exInputs = new float[3][];
//        byte[][] exInputMasks = new byte[2][];

//        Path npFile = Paths.get(getPath("ex.npz"));
//        NpzFile.Reader npzReader = NpzFile.read(npFile);
//        System.out.println(npzReader.introspect());
//        float[] x1 = npzReader.get("0", Integer.MAX_VALUE).asFloatArray();
//        float[] x1f = npzReader.get("1", Integer.MAX_VALUE).asFloatArray();
//        byte[] x1m = npzReader.get("2", Integer.MAX_VALUE).asByteArray();
//        float[] x2 = npzReader.get("3", Integer.MAX_VALUE).asFloatArray();
//        byte[] x2m = npzReader.get("4", Integer.MAX_VALUE).asByteArray();
//        exInputs[0] = x1;
//        exInputs[1] = x1f;
//        exInputs[2] = x2;
//
//        exInputMasks[0] = x1m;
//        exInputMasks[1] = x2m;

//        System.out.println(toString(x1));
//        System.out.println(toString(x1FeedBatch));
//        Files.write(Paths.get("x1np.txt"), toString(x1).getBytes());
//        Files.write(Paths.get("x1FeedBatch.txt"), toString(x1FeedBatch).getBytes());

//        System.out.println(Arrays.toString(x1f));
//        System.out.println(Arrays.toString(x1m));
//        System.out.println(Arrays.toString(x2));
//        System.out.println(Arrays.toString(x2m));

        float[][] out = predict(inputs, inputMasks, batch);
//        float[][] out = predict(exInputs, exInputMasks, batch);

//        System.out.println(Arrays.deepToString(out));

        long end = System.currentTimeMillis();
        System.out.println("time: " + (end - start) + " ms");
    }

    private static String toString(float[] a, int... dims) {
        if (a == null)
            return "null";

        int iMax = a.length - 1;
        if (iMax == -1)
            return "[]";

        StringBuilder b = new StringBuilder();
        b.append('[');
        for (int i = 0; ; i++) {
            if (Math.signum(a[i]) != 0) {
                String s = String.format("%.8f", a[i]);
                b.append(s);
            }else{
                b.append(0);
            }
            if (i == iMax)
                return b.append(']').toString();
            b.append(", ");
        }
    }

    private float[][] predict(float[][] data, byte[][] mask, int batch) {
        int numFeatures = 3;
        int vectorDim = 300;
        byte[] graphDef = readAllBytesOrExit(Paths.get(getPath("tf_reader.pb")));
        Graph g = new Graph();
        assert graphDef != null;
        g.importGraphDef(graphDef);
        Session s = new Session(g);

        Session.Runner runner = s.runner();

//        float[] out = new float[2 * data[0].length / vectorDim];
//        mTFInference.feed("input_1", data[0], batch, data[0].length / batch / vectorDim, vectorDim);
//        mTFInference.feed("input_2", data[1], batch, data[1].length / batch / numFeatures, numFeatures);
//        mTFInference.feed("input_3", mask[0], batch, mask[0].length / batch);
//        mTFInference.feed("input_4", data[2], batch, data[2].length / batch / vectorDim, vectorDim);
//        mTFInference.feed("input_5", mask[1], batch, mask[1].length / batch);
//        mTFInference.run(new String[]{"answer"});
//        mTFInference.fetch("answer", out);

        Tensor<Float> input1 = Tensor.create(new long[]{batch, data[0].length / batch / vectorDim, vectorDim},
                FloatBuffer.wrap(data[0]));
        Tensor<Float> input2 = Tensor.create(new long[]{batch, data[1].length / batch / numFeatures, numFeatures},
                FloatBuffer.wrap(data[1]));
        Tensor<UInt8> input3 = Tensor.create(UInt8.class, new long[]{batch, mask[0].length / batch},
                ByteBuffer.wrap(mask[0]));
        Tensor<Float> input4 = Tensor.create(new long[]{batch, data[2].length / batch / vectorDim, vectorDim},
                FloatBuffer.wrap(data[2]));

        Tensor<Float> result = runner.feed("para/emb", input1)
                .feed("para/feature", input2)
                .feed("para/mask", input3)
                .feed("q_emb", input4)
                .fetch("answer/scores").run().get(0).expect(Float.class);

        final long[] resultShape = result.shape();
        System.out.println(Arrays.toString(resultShape));
        float[][] out = result.copyTo(new float[2][data[0].length / vectorDim]);
        System.out.println(toString(out[0]));
        System.out.println(toString(out[1]));

//        float[][] answers = DecodeUtil.decode(out, batch, data[0].length / batch / vectorDim);
//        System.out.println(Arrays.deepToString(answers));
        return out;
    }

    private static byte[] readAllBytesOrExit(Path path) {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(1);
        }
        return null;
    }


    public static void main(String[] args) throws IOException {
        new RnnReader().benchmark();
    }
}
