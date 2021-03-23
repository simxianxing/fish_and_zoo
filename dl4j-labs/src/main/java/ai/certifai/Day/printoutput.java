package ai.certifai.Day;

import java.util.ArrayList;



public class printoutput {
    private static ArrayList<String> outlist = new ArrayList<>();
    private static ArrayList<String> finallist = new ArrayList<>();
    private static String label;
    private static boolean startnewchar;
    private static String alphabet;
    private static char predicted;


    public static void getAlphabet(String labelA){

        label = labelA;
        outlist.add(label);

        if (label.equals("0")){
            finallist.add(" ");
        }
        if (label.equals("3")){
            startnewchar = true;
            outlist.clear();
        }
        if (startnewchar && label.equals("5")){
            if (!outlist.get(0).equals("5")){
                alphabet = outlist.get(0);
                for (int i = 0; i < outlist.size() - 1; i++) {
                    if (outlist.get(i) != outlist.get(i + 1)){
                        if (!outlist.get(i + 1).equals("4") && !outlist.get(i + 1).equals("5")){
                            alphabet = alphabet + outlist.get(i + 1);
                        }
                    }
                }
                //System.out.println(alphabet);

                switch (alphabet){
                    case "12":
                        //finallist.add("A");
                        predicted = 'A';
                        break;
                    case "2111":
                        //finallist.add("B");
                        predicted = 'B';
                        break;
                    case "2121":
                        //finallist.add("C");
                        predicted = 'C';
                        break;
                    case "211":
                        //finallist.add("D");
                        predicted = 'D';
                        break;
                    case "1":
                        //finallist.add("E");
                        predicted = 'E';
                        break;
                    case "1121":
                        //finallist.add("F");
                        predicted = 'F';
                        break;
                    case "221":
                        //finallist.add("G");
                        predicted = 'G';
                        break;
                    case "1111":
                        //finallist.add("H");
                        predicted = 'H';
                        break;
                    case "11":
                        //finallist.add("I");
                        predicted = 'I';
                        break;
                    case "1222":
                        //finallist.add("J");
                        predicted = 'J';
                        break;
                    case "212":
                        //finallist.add("K");
                        predicted = 'K';
                        break;
                    case "1211":
                        //finallist.add("L");
                        predicted = 'L';
                        break;
                    case "22":
                        //finallist.add("M");
                        predicted = 'M';
                        break;
                    case "21":
                        //finallist.add("N");
                        predicted = 'N';
                        break;
                    case "222":
                        //finallist.add("O");
                        predicted = 'O';
                        break;
                    case "1221":
                        //finallist.add("P");
                        predicted = 'P';
                        break;
                    case "2212":
                        //finallist.add("Q");
                        predicted = 'Q';
                        break;
                    case "121":
                        //finallist.add("R");
                        predicted = 'R';
                        break;
                    case "111":
                        //finallist.add("S");
                        predicted = 'S';
                        break;
                    case "2":
                        //finallist.add("T");
                        predicted = 'T';
                        break;
                    case "112":
                        //finallist.add("U");
                        predicted = 'U';
                        break;
                    case "1112":
                        //finallist.add("V");
                        predicted = 'V';
                        break;
                    case "122":
                        //finallist.add("W");
                        predicted = 'W';
                        break;
                    case "2112":
                        //finallist.add("X");
                        predicted = 'X';
                        break;
                    case "2122":
                        //finallist.add("Y");
                        predicted = 'Y';
                        break;
                    case "2211":
                        //finallist.add("Z");
                        predicted = 'Z';
                        break;
                }
                outlist.clear();
                //System.out.println(finallist.toString());

                System.out.print(predicted);
            }

        }
        if (label.equals("1") || label.equals("2")){
            outlist.add(label);
        }
    }
}




