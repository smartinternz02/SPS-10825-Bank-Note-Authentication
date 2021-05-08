package org.ml;
import java.io.IOException;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Linearreg {
		public static void main(String[] args) throws Exception {
			DataSource source =new DataSource("C:\\\\Users\\\\SWETHA\\\\OneDrive\\\\Desktop\\\\final eclipse3\\\\org.ml\\\\src\\\\main\\\\java\\\\org\\\\ml\\\\data_banknote_authentication (1).csv");
			Instances dataset=source.getDataSet();
			dataset.setClassIndex(dataset.numAttributes()-1);
			//linear Regression
			LinearRegression lr=new LinearRegression();
			lr.buildClassifier(dataset);
			
			Evaluation lreval =new Evaluation(dataset);
		    lreval.evaluateModel(lr,dataset);
			System.out.println(lreval.toSummaryString());
			
			
		}

	}