package test;
import java.io.File;
import java.io.FileWriter;
// import java.util.Arrays;
import java.util.Scanner;

public class TestSolution {
	private static final String CSV_1_PATH = "../resources/1.csv";
	private static final String CSV_2_PATH = "../resources/2.csv";
	private static final String OUTPUT_CSV_PATH = "../resources/my_output.csv";

	public void combineFiles(File csv1, File csv2, File outputFile) throws Exception {
		System.out.println("====.........Started reading the files .....=====");
       	Scanner dataReader = new Scanner(csv1); 
		String file1Data = ""; 
		while (dataReader.hasNextLine()) {  
			file1Data += dataReader.nextLine() + "/n/n";  
		}  
		file1Data.trim();
		dataReader.close();  
		String[] file1arr = file1Data.split("/n/n", -1);
		
		String file2Data = ""; 
		Scanner dataReader2 = new Scanner(csv2); 
		while (dataReader2.hasNextLine()) {  
			file2Data += dataReader2.nextLine() + "/n/n";  
		}  
		dataReader2.close();
		file2Data.trim();  
		String[] file2arr = file2Data.split("/n/n", -1);
		System.out.println("======........Completed reading the files..........=======");

		System.out.println("=======..........Started Processing............======");
		String output = "";
		for (String i: file1arr){
			String second_order_i = i.split(",",-1)[0];
			for (String j: file2arr){
				String second_order_j = j.split(",",-1)[0];
				if (second_order_i.equals(second_order_j)){
					j.replace(second_order_j,"");
					if(second_order_i.equals("") || second_order_j.equals("")){
						continue;
					}
					output += i +","+j+"\n"; //Handling the edgecase
				}
				else{
					continue;
				}
			}
		}
		System.out.println("=======..............Completed Processing.............=========");


		System.out.println("==========.........Started Writing...............=============");
		FileWriter fwrite = new FileWriter(outputFile);  
        fwrite.write(output);   
        fwrite.close();
		System.out.println("===========..............Completed Writing..........============");
	}

	public static void main( String[] args ) {
		try {
			File csv1 = new File( CSV_1_PATH );
			File csv2 = new File( CSV_2_PATH );
			File output = new File( OUTPUT_CSV_PATH );

			TestSolution solution = new TestSolution( );
			solution.combineFiles( csv1, csv2, output );

		} catch ( Exception e ) {
			e.printStackTrace( );
		}
	}
}
