package ai.awk.tools;

import com.google.gson.Gson;
import com.linkedin.paldb.api.PalDB;
import com.linkedin.paldb.api.StoreReader;
import com.linkedin.paldb.api.StoreWriter;
import com.spotify.sparkey.Sparkey;
import com.spotify.sparkey.SparkeyReader;
import com.spotify.sparkey.SparkeyWriter;
import org.apache.commons.io.FilenameUtils;
import org.jetbrains.bio.npy.NpyArray;
import org.jetbrains.bio.npy.NpzFile;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

import static java.util.stream.Collectors.toList;

public class GenDBS {
    private static void printUsage() {
        String usage = "gen emb [embedding file in numpy npz format] [output file]\n"
                + "gen idx [doc folder] [output file]\n"
                + "query emb [padldb database file] [query key]\n"
                + "query idx [padldb database file] [query key]\n";
        System.err.println(usage);
        System.exit(-1);
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            printUsage();
        } else {
            String func = args[0];
            if (func.toLowerCase().startsWith("gen")) {
                if (args[1].equalsIgnoreCase("emb")) {
                    writeEmbDB(args);
                } else if (args[1].equalsIgnoreCase("idx")) {
//                    writeIdxDB(args);
                } else {
                    printUsage();
                }
            } else if (func.toLowerCase().startsWith("query")) {
                if (args[1].equalsIgnoreCase("emb")) {
                    queryEmbDB(args);
                } else if (args[1].equalsIgnoreCase("idx")) {
//                    queryDocDB(args);
                } else {
                    printUsage();
                }
            } else {
                System.err.println("invalid args:" + func + "!");
                printUsage();
            }
        }
    }

//    private static void queryDocDB(String[] args) {
//        String docDBPath = args[2];
//        String query = args[3];
//        StoreReader reader = PalDB.createReader(new File(docDBPath));
//        if (query.startsWith("c_")) {
//            int[][] vector = reader.get(query);
//            System.out.println("\n\033[36mResults for \033[1m" + query + "\033[0m: \033[1m");
//            System.out.println(Arrays.deepToString(vector));
//        } else {
//            int[] vector = reader.get(query);
//            System.out.println("\n\033[36mResults for \033[1m" + query + "\033[0m: \033[1m");
//            System.out.println(Arrays.toString(vector));
//        }
//
//    }

//    private static void writeIdxDB(String[] args) throws IOException {
//        String folderPath = args[2];
//        String outFilePath = args[3];
//        List<Path> docFiles = Files.walk(Paths.get(folderPath))
//                .filter(s -> s.toString().endsWith(".json"))
//                .sorted()
//                .collect(toList());
//        System.out.println(docFiles);
//        Gson gson = new Gson();
//        File dbFile = new File(outFilePath);
//        StoreWriter writer = PalDB.createWriter(dbFile);
//        for (Path doc : docFiles) {
//            System.out.println(doc);
//            String key = FilenameUtils.getBaseName(doc.toString());
//            String content = new String(Files.readAllBytes(doc));
//            if (key.startsWith("c_")) {
//                int[][] value = gson.fromJson(content, int[][].class);
//                writer.put(key, value);
////                System.out.println(Arrays.deepToString(value));
////                System.out.println("put " + key);
//            } else {
//                int[] value = gson.fromJson(content, int[].class);
//                writer.put(key, value);
////                System.out.println("put " + key);
////                System.out.println(Arrays.toString(value));
//            }
//        }
//        writer.close();
//        System.out.println("finished, doc database is at: \033[36m" + dbFile.toString());
//    }


    private static void queryEmbDB(String[] args) throws IOException {
        String embDbPath = args[2];
        String query = args[3];
//        StoreReader reader = PalDB.createReader(new File(embDbPath));
        SparkeyReader reader = Sparkey.open(new File(embDbPath));
        reader.getAsString(query);
        float[] vector = reader.get(Integer.parseInt(query));
        System.out.println("\n\033[36mResults for \033[1m" + query + "\033[0m: \033[1m");
        System.out.println(Arrays.toString(vector));
    }

    private static void writeEmbDB(String[] args) throws IOException {
        String embFilePath = args[2];
        String outFilePath = args[3];

        Path npFile = Paths.get(embFilePath);
        NpzFile.Reader npzReader = NpzFile.read(npFile);
        System.out.println(npzReader.introspect());
        NpyArray embArray = npzReader.get("emb", Integer.MAX_VALUE);
        float[] embeddings = embArray.asFloatArray();
        int[] embShape = embArray.getShape();
//        assert embShape[0] == 214391;
//        assert embShape[1] == 300;
        int embRecords = embShape[0];
        int vectorDim = embShape[1];

        File dbFile = new File(outFilePath);
        SparkeyWriter writer = Sparkey.createNew(dbFile);

//        StoreWriter writer = PalDB.createWriter(dbFile);
        float[] vector = new float[vectorDim];

        for (int i = 0; i < embRecords; i++) {
            System.arraycopy(embeddings, i * vectorDim, vector, 0, vectorDim);
            writer.put(i);
            writer.put(i, vector);
//            System.out.println("put " + (i + 1) + " vector: " + ind2tokens.get(i));
        }
        writer.close();
        System.out.println("finished, emb database is at: \033[36m" + dbFile.toString());
    }

    private static float[] toFloatArray(byte[] bytes) {
        ByteBuffer buffer = ByteBuffer.wrap(bytes);
        FloatBuffer fb = buffer.asFloatBuffer();

        float[] floatArray = new float[fb.limit()];
        fb.get(floatArray);


        return floatArray;
    }
}
