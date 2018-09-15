package ai.awk.tools;

import com.linkedin.paldb.api.PalDB;
import com.linkedin.paldb.api.StoreReader;
import com.linkedin.paldb.api.StoreWriter;
import org.jetbrains.bio.npy.NpyArray;
import org.jetbrains.bio.npy.NpzFile;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

public class BuildEmb {
    private static void printUsage() {
        String usage = "gen [embedding file in numpy npz format] [output file] \n"
                + "get [padldb database file] [query key]\n";
        System.err.println(usage);
        System.exit(-1);
    }

    public static void main(String[] args) {
        if (args.length < 1) {
            printUsage();
        } else {
            String func = args[0];
            if (func.toLowerCase().startsWith("gen")) {
                writeEmbDB(args);
            } else if (func.toLowerCase().startsWith("get")) {
                queryEmbDB(args);
            } else {
                System.err.println("invalid args:" + func + "!");
                printUsage();
            }
        }
    }

    private static void queryEmbDB(String[] args) {
        String embDbPath = args[1];
        String query = args[2];
        StoreReader reader = PalDB.createReader(new File(embDbPath));
        float[] vector = reader.get(Integer.parseInt(query));
        System.out.println("\n\033[36mResults: \033[0m\033[1m");
        System.out.println(Arrays.toString(vector));
    }

    private static void writeEmbDB(String[] args) {
        String embFilePath = args[1];
        String outFilePath = args[2];

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
        StoreWriter writer = PalDB.createWriter(dbFile);
        float[] vector = new float[vectorDim];

        for (int i = 0; i < embRecords; i++) {
            System.arraycopy(embeddings, i * vectorDim, vector, 0, vectorDim);
            writer.put(i, vector);
//            System.out.println("put " + (i + 1) + " vector: " + ind2tokens.get(i));
        }
        writer.close();
        System.out.println("finished, emb database is at: \033[36m" + dbFile.toString());
    }
}
