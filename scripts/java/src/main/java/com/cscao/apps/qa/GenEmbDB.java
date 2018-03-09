package com.cscao.apps.qa;

import com.linkedin.paldb.api.PalDB;
import com.linkedin.paldb.api.StoreReader;
import com.linkedin.paldb.api.StoreWriter;
import org.jetbrains.bio.npy.NpyArray;
import org.jetbrains.bio.npy.NpzFile;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class GenEmbDB {

    public void writeTokenKeyEmbDB() throws IOException {
        String exFile = getClass().getClassLoader().getResource("emb.npz").getPath();
        Path npFile = Paths.get(exFile);
        NpzFile.Reader npzReader = NpzFile.read(npFile);
        System.out.println(npzReader.introspect());
        NpyArray embArray = npzReader.get("emb", Integer.MAX_VALUE);
        float[] embeddings = embArray.asFloatArray();
        int[] embShape = embArray.getShape();
        assert embShape[0] == 214391;
        assert embShape[1] == 300;
        int embRecords = embShape[0];
        int vectorDim = embShape[1];
        StoreWriter writer = PalDB.createWriter(new File("emb.pdb"));
        float[] vector = new float[vectorDim];
        String ind2tokFile = getClass().getClassLoader().getResource("ind2tok.txt").getPath();
        Stream<String> ind2tokStream = Files.lines(Paths.get(ind2tokFile));
        List<String> ind2tokens = ind2tokStream.collect(Collectors.toList());
        for (int i = 0; i < embRecords; i++) {
            System.arraycopy(embeddings, i * vectorDim, vector, 0, vectorDim);
            writer.put(ind2tokens.get(i), vector);
            System.out.println("put " + (i + 1) + " vector: " + ind2tokens.get(i));
        }
        writer.close();
        Iterable<String> iterable = Files.lines(Paths.get(ind2tokFile))::iterator;
        for (String str : iterable) {
            System.out.println(str);
        }

    }

    public void readEmbDb() {
        StoreReader reader = PalDB.createReader(new File("emb.pdb"));
        float[] vector = reader.get("How");
        System.out.println(Arrays.toString(vector));
    }

    public static void main(String[] args) throws IOException {
        new GenEmbDB().writeTokenKeyEmbDB();
        new GenEmbDB().readEmbDb();
    }
}
