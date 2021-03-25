package ai.certifai.Day;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;

public class printoutput {
    private static ArrayList<String> outlist = new ArrayList<>();
    private static String label;
    private static boolean startnewchar, spacedone;
    private static String alphabet;
    private static char predicted, character;
    private static Container cp;
    private static JTextArea text;
    private static int textlength = 27;
    private static int count = 1;

    public static void getAlphabet(String labelA, Container cpA, JTextArea textA){

        label = labelA; //recall predicted output from drawResults
        outlist.add(label); //start append to an empty ArrayList

        //label 4 = Spacing
        if (spacedone && label.equals("4")){
            predicted = ' '; //final prediction
            spacedone = false; //to avoid multiple spacing
            getWindow(predicted, cpA, textA); //feed the final prediction to GUI
        }
        //label 3 = Start new character
        if (label.equals("3")){
            startnewchar = true; //start new character
            spacedone = true; //enable to get space
            outlist.clear(); //clear ArrayList to start new character
        }
        //label 5 = Stop and predict alphabet
        if (startnewchar && label.equals("5")){
            spacedone = false; //stop spacing
            startnewchar = false; //stop output
            if (!outlist.get(0).equals("5")){ // to avoid multiple stop (detection)
                alphabet = outlist.get(0); //store the first dot/dash
                for (int i = 0; i < outlist.size() - 1; i++) { //go through all the outputs stored
                    if (outlist.get(i) != outlist.get(i + 1)){ //compare outputs alternately
                        if (!outlist.get(i + 1).equals("0") && !outlist.get(i + 1).equals("5")){ //remove the break signal (0) and stop signal (5)
                            alphabet = alphabet + outlist.get(i + 1); //create the dot-dash combination
                        }
                    }
                }
                //compare dot-dash
                switch (alphabet){
                    case "12":
                        predicted = 'A'; //final prediction
                        break; //break the case
                    case "2111":
                        predicted = 'B'; //final prediction
                        break; //break the case
                    case "2121":
                        predicted = 'C'; //final prediction
                        break; //break the case
                    case "211":
                        predicted = 'D'; //final prediction
                        break; //break the case
                    case "1":
                        predicted = 'E'; //final prediction
                        break; //break the case
                    case "1121":
                        predicted = 'F'; //final prediction
                        break; //break the case
                    case "221":
                        predicted = 'G'; //final prediction
                        break; //break the case
                    case "1111":
                        predicted = 'H'; //final prediction
                        break; //break the case
                    case "11":
                        predicted = 'I'; //final prediction
                        break; //break the case
                    case "1222":
                        predicted = 'J'; //final prediction
                        break; //break the case
                    case "212":
                        predicted = 'K'; //final prediction
                        break; //break the case
                    case "1211":
                        predicted = 'L'; //final prediction
                        break; //break the case
                    case "22":
                        predicted = 'M'; //final prediction
                        break; //break the case
                    case "21":
                        predicted = 'N'; //final prediction
                        break; //break the case
                    case "222":
                        predicted = 'O'; //final prediction
                        break; //break the case
                    case "1221":
                        predicted = 'P'; //final prediction
                        break; //break the case
                    case "2212":
                        predicted = 'Q'; //final prediction
                        break; //break the case
                    case "121":
                        predicted = 'R'; //final prediction
                        break; //break the case
                    case "111":
                        predicted = 'S'; //final prediction
                        break; //break the case
                    case "2":
                        predicted = 'T'; //final prediction
                        break; //break the case
                    case "112":
                        predicted = 'U'; //final prediction
                        break; //break the case
                    case "1112":
                        predicted = 'V'; //final prediction
                        break; //break the case
                    case "122":
                        predicted = 'W'; //final prediction
                        break; //break the case
                    case "2112":
                        predicted = 'X'; //final prediction
                        break; //break the case
                    case "2122":
                        predicted = 'Y'; //final prediction
                        break; //break the case
                    case "2211":
                        predicted = 'Z'; //final prediction
                        break; //break the case
                }
                outlist.clear(); //clear ArrayList to start new character
                getWindow(predicted, cpA, textA); //feed the final prediction to GUI
            }

        }
        if (label.equals("1") || label.equals("2") || label.equals("0")){
            spacedone = false; //stop spacing
            outlist.add(label); //store dot (1) , dash (2) and break (0)
        }
    }

    public static void getWindow(char characterA, Container cpA, JTextArea textA){

        text = textA; //recall the Text Area (text font, size, colour)
        cp = cpA; //recall the container
        character = characterA; //recall the final prediction (Alphabet)

        if (textlength == count+1){
            System.out.print("\n"); //print new line on terminal
            text.append("\n"); //print new line on Text Area
            cp.add(text); //add to container (window)
            count = 1;
        }
        System.out.print(character); //print final prediction on terminal
        text.append(String.valueOf(character)); //print final prediction on Text Area
        cp.add(text); //add to container (window)
        count++;
    }
}




