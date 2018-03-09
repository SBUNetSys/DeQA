package com.cscao.apps.qa;

import java.util.Arrays;

/***
 * Created by qqcao on Mar 03, 2018.
 *
 */

public class Feature {
    private String[] ner;
    private String[] pos;
    private String[] words;
    private float[] tf;

    public Feature() {
    }

    public Feature(String[] ner, String[] pos, String[] words, float[] tf) {
        this.ner = ner;
        this.pos = pos;
        this.words = words;
        this.tf = tf;
    }

    public float[] getTf() {
        return tf;
    }

    public void setTf(float[] tf) {
        this.tf = tf;
    }

    public String[] getNer() {
        return ner;
    }

    public void setNer(String[] ner) {
        this.ner = ner;
    }

    public String[] getPos() {
        return pos;
    }

    public void setPos(String[] pos) {
        this.pos = pos;
    }

    public String[] getWords() {
        return words;
    }

    public void setWords(String[] words) {
        this.words = words;
    }

    @Override
    public String toString() {
        return "Feature{" +
                "ner=" + Arrays.toString(ner) +
                ", pos=" + Arrays.toString(pos) +
                ", words=" + Arrays.toString(words) +
                ", tf=" + Arrays.toString(tf) +
                '}';
    }
}
