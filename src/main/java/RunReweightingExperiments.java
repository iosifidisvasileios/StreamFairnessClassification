
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.Classifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.InstanceExample;
import moa.streams.ArffFileStream;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

//import moa.evaluation.WindowAUCImbalancedPerformanceEvaluator;

public class RunReweightingExperiments {
	protected static int saPos = 0, saNeg=0, nSaPos = 0, nSaNeg=0; //number of true values
	protected static int tpDeprived=0, fpDeprived=0, tpFavored=0, fpFavored=0;
	protected static int tnDeprived=0, fnDeprived=0, tnFavored=0, fnFavored=0;

	protected static double favPos=0, favNeg=0, savPos=0, savNeg=0;//weights 
	protected static int epsilon = 0; //percentage
	//data definition
	protected static int saIndex=0;
	protected static String saVal = "Female";
	protected static int desiredClass = 1;
	protected static int notDesiredClass = 0;

	protected static String saName = "sex";
	protected static int windowSize=0;
	protected static InstanceExample[] windowList = new InstanceExample[windowSize];
	//different classifiers
	protected static Classifier learner;

	public static void main(String[] args) throws FileNotFoundException {
		saName = "sex";
		saVal = "Female";

		String outfile ="";
		String infile ="";
//
//		if(args.length != 4){
//			System.out.println("1:B/S/G 2:epsilon_int 3:window_size_int 4:NB/HT/KNN/AUE");
//			exit(1);
//		}
//		epsilon= Integer.valueOf(args[1]);
//		windowSize 	= Integer.valueOf(args[2]);
//
//		if(args[0].equals("B")) {
//			infile = "/home/iosifidis/HansCode/big_remove_InstanceWeight_shuffling1.arff";
//			outfile = "/home/iosifidis/HansCode/RW_RESULTS/BigDataset/epsilon_" + epsilon + "/Classifier_" + args[3] +"/window_" + windowSize + "/";
//		}else if(args[0].equals("S")){
//			infile="/home/iosifidis/HansCode/small_dataset_remove_fnlwgt_shuffling1.arff";
//			outfile = "/home/iosifidis/HansCode/RW_RESULTS/SmallDataset/epsilon_" + epsilon + "/Classifier_" + args[3]+ "/window_" + windowSize + "/";
//		} else if (args[0].equals("G")){
//			saIndex=2;
//			desiredClass = 1;
//			notDesiredClass = 0;
//			saName = "SA";
//			saVal = "Female";
//
//			infile = "/home/iosifidis/HansCode/synthetic_data.arff";
//			outfile ="/home/iosifidis/HansCode/RW_RESULTS/StreamGenerator/epsilon_" + epsilon + "/Classifier_" + args[3]+ "/window_" + windowSize + "/";
//		} else {
//			exit(1);
//		}
//
//		new File(outfile).mkdirs();
//
//		if(args[3].equals("NB")){
//			learner = new NaiveBayes();
//		} else if (args[3].equals("HT")){
//			learner = new HoeffdingTree();
//		} else if (args[3].equals("KNN")){
//			learner = new kNN();//5.
//		} else if (args[3].equals("AUE")){
//			learner = new AccuracyUpdatedEnsemble();
//		} else{
//			exit(1);
//		}




		epsilon= 0;
		windowSize 	= 1000;

//        saIndex=2;
//        desiredClass = 1;
//        notDesiredClass = 0;
//        saName = "SA";
//        saVal = "Female";
//
		infile = "/home/iosifidis/HansCode/big_remove_InstanceWeight_shuffling1.arff";
// 		String infileWithoutSA = "/home/iosifidis/HansCode/synthetic_data_NOSA.arff";

        outfile ="/home/iosifidis/HansCode/ValidationTest/";
        learner = new HoeffdingTree();//5.
		new File(outfile).mkdirs();


		PrintWriter writer = new PrintWriter(outfile + "time.csv");

		long startTime = System.currentTimeMillis();
		long endTime = System.currentTimeMillis();



//		CreateNewClassifierThenTrainWithSameWeights(infile,outfile + "ResetAndTrainSameW.csv");//1.
//		endTime = System.currentTimeMillis();
//		writer.println("ResetAndTrainSameW:" + (endTime - startTime));
//
//		learner.resetLearning();

//		startTime = System.currentTimeMillis();
//		CreateNewClassifierThenTrainWithNewWeights(infile,outfile + "ResetAndTrainNewW.csv");//2.
//		endTime = System.currentTimeMillis();
//		writer.println("ResetAndTrainNewW:" + (endTime - startTime));
//
//		learner.resetLearning();

//		startTime = System.currentTimeMillis();
//		CreateNewClassifierNoTrain(infile,outfile+"ResetAndNoTrain.csv");//3.
//		endTime = System.currentTimeMillis();
//		writer.println("ResetAndNoTrain:" + (endTime - startTime));
//
//		learner.resetLearning();

//		startTime = System.currentTimeMillis();
//		AccClassifierThenTrainWithSameWeights(infile,outfile + "AccumAndTrainSameW.csv");//4.
//		endTime = System.currentTimeMillis();
//		writer.println("AccumAndTrainSameW:" + (endTime - startTime));

//		learner.resetLearning();
//		startTime = System.currentTimeMillis();
//		AccClassifierThenTrainWithNewWeights(infile,outfile + "AccumAndTrainNewW.csv");//5.
//		endTime = System.currentTimeMillis();
//		writer.println("AccumAndTrainNewW:" + (endTime - startTime));
//
//		learner.resetLearning();

		startTime = System.currentTimeMillis();
		AccClassifierNoTrain(infile,outfile + "AccumAndNoTrain.csv");//6.
		endTime = System.currentTimeMillis();
		writer.println("AccumAndNoTrain:" + (endTime - startTime));

//		learner.resetLearning();
//
//		startTime = System.currentTimeMillis();
//		CreateNewClassifierThenTrainWithSameWeights_PredictNoSA(infile,outfile+ "ResetAndTrainSameWNoSA.csv");//6.
//		endTime = System.currentTimeMillis();
//		writer.println("ResetAndTrainSameWNoSA:" + (endTime - startTime));
//
//		learner.resetLearning();
//
//		startTime = System.currentTimeMillis();
//		CreateNewClassifierThenTrainWithNewWeights_PredictNoSA(infile,outfile+ "ResetAndTrainNewWNoSA.csv");//8.
//		endTime = System.currentTimeMillis();
//		writer.println("ResetAndTrainNewWNoSA:" + (endTime - startTime));
//
//		learner.resetLearning();
//
//		startTime = System.currentTimeMillis();
//		CreateNewClassifierNoTrain_PredictNoSA(infile,outfile+ "ResetAndNoTrainNoSA.csv");//9.
//		endTime = System.currentTimeMillis();
//		writer.println("ResetAndNoTrainNoSA:" + (endTime - startTime));
//
//		learner.resetLearning();
//
//		startTime = System.currentTimeMillis();
//		AccClassifierThenTrainWithSameWeights_PredictNoSA(infile,outfile + "AccumAndTrainSameWNoSA.csv");//10.
//		endTime = System.currentTimeMillis();
//		writer.println("AccumAndTrainSameWNoSA:" + (endTime - startTime));
//
//		learner.resetLearning();
//
//		startTime = System.currentTimeMillis();
//		AccClassifierThenTrainWithNewWeights_PredictNoSA(infile,outfile + "AccumAndTrainNewWNoSA.csv");//11.
//		endTime = System.currentTimeMillis();
//		writer.println("AccumAndTrainNewWNoSA:" + (endTime - startTime));

		learner.resetLearning();
		startTime = System.currentTimeMillis();
		AccClassifierNoTrain_PredictNoSA(infile,outfile + "AccumAndNoTrainNoSA.csv");//12.
		endTime = System.currentTimeMillis();
		writer.println("AccumAndNoTrainNoSA:" + (endTime - startTime));
		writer.close();

		learner.resetLearning();
	}


	//model 1
	//ResetLearner:x	PredictNoSA		train_same_weights:x		train_new_weights
	public static void CreateNewClassifierThenTrainWithSameWeights(String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		windowList = new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false; int numOfChanges = 0;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;

			numberSamples++;
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);
			if (stopTrain==false)//training from beginning until bias occurs
				learner.trainOnInstance(trainInst);

			if (numberSamples>=windowSize){
				if (numberSamples%windowSize==0){
					cnt++;
					//for calculation of discrimination score
					String[] labels = evaluator.getAucEstimator().getSAVal();
					double[] predictions = evaluator.getAucEstimator().getPredictions();
					int[] trueLabels = evaluator.getAucEstimator().getTrueLabel();
					double discClassifier=DiscriminationScore(labels,predictions,trueLabels);
					avgDisc+=Math.abs(discClassifier);
					double discData = Disc_Data();
					if (Math.abs(discClassifier)>epsilon){ numOfChanges +=1;
						learner.resetLearning();
						stopTrain=true;
						ApplyReweighing();
					}
					for (int i=0; i<windowSize; i++)
						learner.trainOnInstance(windowList[i]);
					//evaluation
					double precision= evaluator.getAucEstimator().getPrecision();
					double acc = evaluator.getAucEstimator().getAccuracy();
					double recall=evaluator.getAucEstimator().getRecall();
					double f1=0;
					if (precision!=0||recall!=0)
						f1=2*precision*recall/(precision+recall);
					double rocArea= evaluator.getAucEstimator().getAUC();
					double prArea=evaluator.getAucEstimator().prArea();
					avgAcc+=acc;
					avgF1+=f1;
					avgROC+=rocArea;
					avgPR+=prArea;
					double kappa=evaluator.getAucEstimator().getKappa();
					double kappaM=evaluator.getAucEstimator().getKappaM();
					double gMean=evaluator.getAucEstimator().getGMean();
					avgKappa+=kappa;
					avgKappaM+=kappaM;
					avgGMean+=gMean;
					writer.println(numberSamples+","+discData+","+discClassifier+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);

				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numOfChanges);
		writer.close();
		System.out.println("Done 1.");
	}
	//model 2
	//ResetLearner:x	PredictNoSA		train_same_weights			train_new_weights:x
	public static void CreateNewClassifierThenTrainWithNewWeights(String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		windowList = new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false; int numOfChanges = 0;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;

			numberSamples++;
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);
			if (stopTrain==false)//training from beginning until bias occurs
				learner.trainOnInstance(trainInst);

			if (numberSamples>=windowSize){
				if (numberSamples%windowSize==0){
					cnt++;
					//for calculation of discrimination score
					String[] labels = evaluator.getAucEstimator().getSAVal();
					double[] predictions = evaluator.getAucEstimator().getPredictions();
					int[] trueLabels = evaluator.getAucEstimator().getTrueLabel();
					double discClassifier=DiscriminationScore(labels,predictions,trueLabels);
					avgDisc+=Math.abs(discClassifier);
					double discData = Disc_Data();
					if (Math.abs(discClassifier)>epsilon){ numOfChanges +=1;
						learner.resetLearning();
						stopTrain=true;
						ApplyReweighing();
					}else{//no disc in the window -> train with new weight
						if (favPos!=0||favNeg!=0||savPos!=0||savNeg!=0){
							for (int i=0; i<windowSize; i++){
								String[] splits=windowList[i].instance.toString().split(",");
								int cl=Integer.parseInt(splits[splits.length-1]);
								if (splits[saIndex].equals(saVal)){//Deprived					
									if (cl==desiredClass)//Positive class
										windowList[i].instance.setWeight(savPos);
									else
										windowList[i].instance.setWeight(savNeg);
								}else{
									if (cl==desiredClass)//Positive class
										windowList[i].instance.setWeight(favPos);
									else
										windowList[i].instance.setWeight(favNeg);
								}
							}
						}
					}
					for (int i=0; i<windowSize; i++)
						learner.trainOnInstance(windowList[i]);
					//evaluation
					double precision= evaluator.getAucEstimator().getPrecision();
					double acc = evaluator.getAucEstimator().getAccuracy();
					double recall=evaluator.getAucEstimator().getRecall();
					double f1=0;
					if (precision!=0||recall!=0)
						f1=2*precision*recall/(precision+recall);
					double rocArea= evaluator.getAucEstimator().getAUC();
					double prArea=evaluator.getAucEstimator().prArea();
					avgAcc+=acc;
					avgF1+=f1;
					avgROC+=rocArea;
					avgPR+=prArea;
					double kappa=evaluator.getAucEstimator().getKappa();
					double kappaM=evaluator.getAucEstimator().getKappaM();
					double gMean=evaluator.getAucEstimator().getGMean();
					avgKappa+=kappa;
					avgKappaM+=kappaM;
					avgGMean+=gMean;
					writer.println(numberSamples+","+discData+","+discClassifier+","+acc+","+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);

				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numOfChanges);
		writer.close();
		System.out.println("Done 2.");
	}
	//model 3
	//ResetLearner:x	PredictNoSA		train_same_weights			train_new_weights
	public static void CreateNewClassifierNoTrain(String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		windowList = new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false; int numOfChanges = 0;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;

			numberSamples++;
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);
			if (stopTrain==false)//training from beginning until bias occurs
				learner.trainOnInstance(trainInst);

			if (numberSamples>=windowSize){
				if (numberSamples%windowSize==0){
					cnt++;
					//for calculation of discrimination score
					String[] labels = evaluator.getAucEstimator().getSAVal();
					double[] predictions = evaluator.getAucEstimator().getPredictions();
					int[] trueLabels = evaluator.getAucEstimator().getTrueLabel();
					double discClassifier=DiscriminationScore(labels,predictions,trueLabels);
					avgDisc+=Math.abs(discClassifier);
					double discData = Disc_Data();
					if (Math.abs(discClassifier)>epsilon){ numOfChanges +=1;
						learner.resetLearning();
						stopTrain=true;

						learner.resetLearning();
						ApplyReweighing();
						for (int i=0; i<windowSize; i++)
							learner.trainOnInstance(windowList[i]);
					}
					//evaluation
					double precision= evaluator.getAucEstimator().getPrecision();
					double acc = evaluator.getAucEstimator().getAccuracy();
					double recall=evaluator.getAucEstimator().getRecall();
					double f1=0;
					if (precision!=0||recall!=0)
						f1=2*precision*recall/(precision+recall);
					double rocArea= evaluator.getAucEstimator().getAUC();
					double prArea=evaluator.getAucEstimator().prArea();
					avgAcc+=acc;
					avgF1+=f1;
					avgROC+=rocArea;
					avgPR+=prArea;
					double kappa=evaluator.getAucEstimator().getKappa();
					double kappaM=evaluator.getAucEstimator().getKappaM();
					double gMean=evaluator.getAucEstimator().getGMean();
					avgKappa+=kappa;
					avgKappaM+=kappaM;
					avgGMean+=gMean;
					writer.println(numberSamples+","+discData+","+discClassifier+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);

				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numOfChanges);
		writer.close();
		System.out.println("Done 3.");
	}
	//model 4
	//ResetLearner		PredictNoSA		train_same_weights:x		train_new_weights
	public static void AccClassifierThenTrainWithSameWeights(String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, savPos,saPos,savNeg,saNeg,favPos,nSaPos,favNeg,nSaNeg,DiscData, DiscClassifier,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		windowList = new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false; int numOfChanges = 0;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;

			numberSamples++;
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);
			if (stopTrain==false)//training from beginning until bias occurs
				learner.trainOnInstance(trainInst);

			if (numberSamples>=windowSize){
				if (numberSamples%windowSize==0){
					cnt++;
					//for calculation of discrimination score
					String[] labels = evaluator.getAucEstimator().getSAVal();
					double[] predictions = evaluator.getAucEstimator().getPredictions();
					int[] trueLabels = evaluator.getAucEstimator().getTrueLabel();
					double discClassifier=DiscriminationScore(labels,predictions,trueLabels);
					avgDisc+=Math.abs(discClassifier);
					double discData = Disc_Data();
					if (Math.abs(discClassifier)>epsilon){ numOfChanges +=1;
						if (stopTrain==false){
							stopTrain=true;
							learner.resetLearning();
						}
						ApplyReweighing();
					}
					for (int i=0; i<windowSize; i++)
						learner.trainOnInstance(windowList[i]);
					//evaluation
					double precision= evaluator.getAucEstimator().getPrecision();
					double acc = evaluator.getAucEstimator().getAccuracy();
					double recall=evaluator.getAucEstimator().getRecall();
					double f1=0;
					if (precision!=0||recall!=0)
						f1=2*precision*recall/(precision+recall);
					double rocArea= evaluator.getAucEstimator().getAUC();
					double prArea=evaluator.getAucEstimator().prArea();
					avgAcc+=acc;
					avgF1+=f1;
					avgROC+=rocArea;
					avgPR+=prArea;
					double kappa=evaluator.getAucEstimator().getKappa();
					double kappaM=evaluator.getAucEstimator().getKappaM();
					double gMean=evaluator.getAucEstimator().getGMean();
					avgKappa+=kappa;
					avgKappaM+=kappaM;
					avgGMean+=gMean;
					writer.println(numberSamples+","+savPos+","+saPos+","+savNeg+","+saNeg+","+favPos+","+nSaPos+","+favNeg+","+nSaNeg+","+discData+","+discClassifier+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);

				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numOfChanges);
		writer.close();
		System.out.println("Done 4.");
	}
	//model 5
	//ResetLearner		PredictNoSA		train_same_weights			train_new_weights:x
	public static void AccClassifierThenTrainWithNewWeights(String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		windowList = new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false; int numOfChanges = 0;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;

			numberSamples++;
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);
			if (stopTrain==false)//training from beginning until bias occurs
				learner.trainOnInstance(trainInst);

			if (numberSamples>=windowSize){
				if (numberSamples%windowSize==0){
					cnt++;
					//for calculation of discrimination score
					String[] labels = evaluator.getAucEstimator().getSAVal();
					double[] predictions = evaluator.getAucEstimator().getPredictions();
					int[] trueLabels = evaluator.getAucEstimator().getTrueLabel();
					double discClassifier=DiscriminationScore(labels,predictions,trueLabels);
					avgDisc+=Math.abs(discClassifier);
					double discData = Disc_Data();
					if (Math.abs(discClassifier)>epsilon){ numOfChanges +=1;
						if (stopTrain==false){
							learner.resetLearning();
							stopTrain=true;
						}
						ApplyReweighing();
					}else{//no disc in the window -> train with new weight
						if (favPos!=0||favNeg!=0||savPos!=0||savNeg!=0){
							for (int i=0; i<windowSize; i++){
								String[] splits=windowList[i].instance.toString().split(",");
								int cl=Integer.parseInt(splits[splits.length-1]);
								if (splits[saIndex].equals(saVal)){//Deprived					
									if (cl==desiredClass)//Positive class
										windowList[i].instance.setWeight(savPos);
									else
										windowList[i].instance.setWeight(savNeg);
								}else{
									if (cl==desiredClass)//Positive class
										windowList[i].instance.setWeight(favPos);
									else
										windowList[i].instance.setWeight(favNeg);
								}
							}
						}
					}
					for (int i=0; i<windowSize; i++)
						learner.trainOnInstance(windowList[i]);
					//evaluation
					double precision= evaluator.getAucEstimator().getPrecision();
					double acc = evaluator.getAucEstimator().getAccuracy();
					double recall=evaluator.getAucEstimator().getRecall();
					double f1=0;
					if (precision!=0||recall!=0)
						f1=2*precision*recall/(precision+recall);
					double rocArea= evaluator.getAucEstimator().getAUC();
					double prArea=evaluator.getAucEstimator().prArea();
					avgAcc+=acc;
					avgF1+=f1;
					avgROC+=rocArea;
					avgPR+=prArea;
					double kappa=evaluator.getAucEstimator().getKappa();
					double kappaM=evaluator.getAucEstimator().getKappaM();
					double gMean=evaluator.getAucEstimator().getGMean();
					avgKappa+=kappa;
					avgKappaM+=kappaM;
					avgGMean+=gMean;
					writer.println(numberSamples+","+discData+","+discClassifier+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);

				}
			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numOfChanges);
		writer.close();
		System.out.println("Done 5.");
	}
	//model 6
	//ResetLearner		PredictNoSA		train_same_weights			train_new_weights
	public static void AccClassifierNoTrain(String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		windowList = new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false; int numOfChanges = 0;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;

			numberSamples++;
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);
			if (stopTrain==false)//training from beginning until bias occurs
				learner.trainOnInstance(trainInst);

			if (numberSamples>=windowSize){
				if (numberSamples%windowSize==0){
					cnt++;
					//for calculation of discrimination score
					String[] labels = evaluator.getAucEstimator().getSAVal();
					double[] predictions = evaluator.getAucEstimator().getPredictions();
					int[] trueLabels = evaluator.getAucEstimator().getTrueLabel();

					double discClassifier=DiscriminationScore(labels,predictions,trueLabels);

					avgDisc+=Math.abs(discClassifier);
					double discData = Disc_Data();
					if (Math.abs(discClassifier)>epsilon){ numOfChanges +=1;
						if (stopTrain==false){
							stopTrain=true;
							learner.resetLearning();
						}
						ApplyReweighing();
						for (int i=0; i<windowSize; i++)
							learner.trainOnInstance(windowList[i]);
					}
					//evaluation
					double precision= evaluator.getAucEstimator().getPrecision();
					double acc = evaluator.getAucEstimator().getAccuracy();
					double recall=evaluator.getAucEstimator().getRecall();
					double f1=0;
					if (precision!=0||recall!=0)
						f1=2*precision*recall/(precision+recall);
					double rocArea= evaluator.getAucEstimator().getAUC();
					double prArea=evaluator.getAucEstimator().prArea();
					avgAcc+=acc;
					avgF1+=f1;
					avgROC+=rocArea;
					avgPR+=prArea;
					double kappa=evaluator.getAucEstimator().getKappa();
					double kappaM=evaluator.getAucEstimator().getKappaM();
					double gMean=evaluator.getAucEstimator().getGMean();
					avgKappa+=kappa;
					avgKappaM+=kappaM;
					avgGMean+=gMean;
					writer.println(numberSamples+","+discData+","+discClassifier+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);

				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numOfChanges);
		writer.close();
		System.out.println("Done 6.");
	}

	//model 7
	//ResetLearner:x	PredictNoSA:x	train_same_weights:x		train_new_weights
	public static void CreateNewClassifierThenTrainWithSameWeights_PredictNoSA(String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		windowList = new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false; int numOfChanges = 0;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;

			numberSamples++;
			Instance tmpInst = trainInst.copy();
			tmpInst.setMissing(saIndex);
			double[] votes = learner.getVotesForInstance(tmpInst);
			evaluator.addResult(trainInstanceExample, votes);
			if (stopTrain==false)//training from beginning until bias occurs
				learner.trainOnInstance(trainInst);

			if (numberSamples>=windowSize){
				if (numberSamples%windowSize==0){
					cnt++;
					//for calculation of discrimination score
					String[] labels = evaluator.getAucEstimator().getSAVal();
					double[] predictions = evaluator.getAucEstimator().getPredictions();
					int[] trueLabels = evaluator.getAucEstimator().getTrueLabel();
					double discClassifier=DiscriminationScore(labels,predictions,trueLabels);
					avgDisc+=Math.abs(discClassifier);
					double discData = Disc_Data();
					if (Math.abs(discClassifier)>epsilon){ numOfChanges +=1;
						learner.resetLearning();
						stopTrain=true;
						ApplyReweighing();
					}
					for (int i=0; i<windowSize; i++)
						learner.trainOnInstance(windowList[i]);
					//evaluation
					double precision= evaluator.getAucEstimator().getPrecision();
					double acc = evaluator.getAucEstimator().getAccuracy();
					double recall=evaluator.getAucEstimator().getRecall();
					double f1=0;
					if (precision!=0||recall!=0)
						f1=2*precision*recall/(precision+recall);
					double rocArea= evaluator.getAucEstimator().getAUC();
					double prArea=evaluator.getAucEstimator().prArea();
					avgAcc+=acc;
					avgF1+=f1;
					avgROC+=rocArea;
					avgPR+=prArea;
					double kappa=evaluator.getAucEstimator().getKappa();
					double kappaM=evaluator.getAucEstimator().getKappaM();
					double gMean=evaluator.getAucEstimator().getGMean();
					avgKappa+=kappa;
					avgKappaM+=kappaM;
					avgGMean+=gMean;
					writer.println(numberSamples+","+discData+","+discClassifier+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);

				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numOfChanges);
		writer.close();
		System.out.println("Done 7.");
	}

	//model 8
	//ResetLearner:x	PredictNoSA:x	train_same_weights			train_new_weights:x
	public static void CreateNewClassifierThenTrainWithNewWeights_PredictNoSA(String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		windowList = new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false; int numOfChanges = 0;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;

			numberSamples++;
			Instance tmpInst = trainInst.copy();
			tmpInst.setMissing(saIndex);
			double[] votes = learner.getVotesForInstance(tmpInst);
			evaluator.addResult(trainInstanceExample, votes);
			if (stopTrain==false)//training from beginning until bias occurs
				learner.trainOnInstance(trainInst);

			if (numberSamples>=windowSize){
				if (numberSamples%windowSize==0){
					cnt++;
					//for calculation of discrimination score
					String[] labels = evaluator.getAucEstimator().getSAVal();
					double[] predictions = evaluator.getAucEstimator().getPredictions();
					int[] trueLabels = evaluator.getAucEstimator().getTrueLabel();
					double discClassifier=DiscriminationScore(labels,predictions,trueLabels);
					avgDisc+=Math.abs(discClassifier);
					double discData = Disc_Data();
					if (Math.abs(discClassifier)>epsilon){ numOfChanges +=1;
						learner.resetLearning();
						stopTrain=true;
						ApplyReweighing();
					}else{//no disc in the window -> train with new weight
						if (favPos!=0||favNeg!=0||savPos!=0||savNeg!=0){
							for (int i=0; i<windowSize; i++){
								String[] splits=windowList[i].instance.toString().split(",");
								int cl=Integer.parseInt(splits[splits.length-1]);
								if (splits[saIndex].equals(saVal)){//Deprived					
									if (cl==desiredClass)//Positive class
										windowList[i].instance.setWeight(savPos);
									else
										windowList[i].instance.setWeight(savNeg);
								}else{
									if (cl==desiredClass)//Positive class
										windowList[i].instance.setWeight(favPos);
									else
										windowList[i].instance.setWeight(favNeg);
								}
							}
						}
					}
					for (int i=0; i<windowSize; i++)
						learner.trainOnInstance(windowList[i]);
					//evaluation
					double precision= evaluator.getAucEstimator().getPrecision();
					double acc = evaluator.getAucEstimator().getAccuracy();
					double recall=evaluator.getAucEstimator().getRecall();
					double f1=0;
					if (precision!=0||recall!=0)
						f1=2*precision*recall/(precision+recall);
					double rocArea= evaluator.getAucEstimator().getAUC();
					double prArea=evaluator.getAucEstimator().prArea();
					avgAcc+=acc;
					avgF1+=f1;
					avgROC+=rocArea;
					avgPR+=prArea;
					double kappa=evaluator.getAucEstimator().getKappa();
					double kappaM=evaluator.getAucEstimator().getKappaM();
					double gMean=evaluator.getAucEstimator().getGMean();
					avgKappa+=kappa;
					avgKappaM+=kappaM;
					avgGMean+=gMean;
					writer.println(numberSamples+","+discData+","+discClassifier+","+acc+","+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);

				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numOfChanges);
		writer.close();
		System.out.println("Done 8.");
	}

	//model 9
	//ResetLearner:x	PredictNoSA:x	train_same_weights			train_new_weights
	public static void CreateNewClassifierNoTrain_PredictNoSA(String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		windowList = new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false; int numOfChanges = 0;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;

			numberSamples++;
			Instance tmpInst = trainInst.copy();
			tmpInst.setMissing(saIndex);
			double[] votes = learner.getVotesForInstance(tmpInst);
			evaluator.addResult(trainInstanceExample, votes);
			if (stopTrain==false)//training from beginning until bias occurs
				learner.trainOnInstance(trainInst);

			if (numberSamples>=windowSize){
				if (numberSamples%windowSize==0){
					cnt++;
					//for calculation of discrimination score
					String[] labels = evaluator.getAucEstimator().getSAVal();
					double[] predictions = evaluator.getAucEstimator().getPredictions();
					int[] trueLabels = evaluator.getAucEstimator().getTrueLabel();
					double discClassifier=DiscriminationScore(labels,predictions,trueLabels);
					avgDisc+=Math.abs(discClassifier);
					double discData = Disc_Data();
					if (Math.abs(discClassifier)>epsilon){ numOfChanges +=1;
						learner.resetLearning();
						stopTrain=true;
						learner.resetLearning();
						ApplyReweighing();
						for (int i=0; i<windowSize; i++)
							learner.trainOnInstance(windowList[i]);
					}
					//evaluation
					double precision= evaluator.getAucEstimator().getPrecision();
					double acc = evaluator.getAucEstimator().getAccuracy();
					double recall=evaluator.getAucEstimator().getRecall();
					double f1=0;
					if (precision!=0||recall!=0)
						f1=2*precision*recall/(precision+recall);
					double rocArea= evaluator.getAucEstimator().getAUC();
					double prArea=evaluator.getAucEstimator().prArea();
					avgAcc+=acc;
					avgF1+=f1;
					avgROC+=rocArea;
					avgPR+=prArea;
					double kappa=evaluator.getAucEstimator().getKappa();
					double kappaM=evaluator.getAucEstimator().getKappaM();
					double gMean=evaluator.getAucEstimator().getGMean();
					avgKappa+=kappa;
					avgKappaM+=kappaM;
					avgGMean+=gMean;
					writer.println(numberSamples+","+discData+","+discClassifier+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);

				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numOfChanges);
		writer.close();
		System.out.println("Done 9.");
	}

	//model 10
	//ResetLearner		PredictNoSA:x	train_same_weights:x		train_new_weights
	public static void AccClassifierThenTrainWithSameWeights_PredictNoSA(String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		windowList = new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false; int numOfChanges = 0;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;

			numberSamples++;
			Instance tmpInst = trainInst.copy();
			tmpInst.setMissing(saIndex);
			double[] votes = learner.getVotesForInstance(tmpInst);
			evaluator.addResult(trainInstanceExample, votes);
			if (stopTrain==false)//training from beginning until bias occurs
				learner.trainOnInstance(trainInst);

			if (numberSamples>=windowSize){
				if (numberSamples%windowSize==0){
					cnt++;
					//for calculation of discrimination score
					String[] labels = evaluator.getAucEstimator().getSAVal();
					double[] predictions = evaluator.getAucEstimator().getPredictions();
					int[] trueLabels = evaluator.getAucEstimator().getTrueLabel();
					double discClassifier=DiscriminationScore(labels,predictions,trueLabels);
					avgDisc+=Math.abs(discClassifier);
					double discData = Disc_Data();
					if (Math.abs(discClassifier)>epsilon){ numOfChanges +=1;
						if (stopTrain==false){
							stopTrain=true;
							learner.resetLearning();
						}
						ApplyReweighing();
					}
					for (int i=0; i<windowSize; i++)
						learner.trainOnInstance(windowList[i]);
					//evaluation
					double precision= evaluator.getAucEstimator().getPrecision();
					double acc = evaluator.getAucEstimator().getAccuracy();
					double recall=evaluator.getAucEstimator().getRecall();
					double f1=0;
					if (precision!=0||recall!=0)
						f1=2*precision*recall/(precision+recall);
					double rocArea= evaluator.getAucEstimator().getAUC();
					double prArea=evaluator.getAucEstimator().prArea();
					avgAcc+=acc;
					avgF1+=f1;
					avgROC+=rocArea;
					avgPR+=prArea;
					double kappa=evaluator.getAucEstimator().getKappa();
					double kappaM=evaluator.getAucEstimator().getKappaM();
					double gMean=evaluator.getAucEstimator().getGMean();
					avgKappa+=kappa;
					avgKappaM+=kappaM;
					avgGMean+=gMean;
					writer.println(numberSamples+","+discData+","+discClassifier+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);

				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numOfChanges);
		writer.close();
		System.out.println("Done 10.");
	}
	//model 11
	//ResetLearner		PredictNoSA:x	train_same_weights			train_new_weights:x
	public static void AccClassifierThenTrainWithNewWeights_PredictNoSA(String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		windowList = new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false; int numOfChanges = 0;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;

			numberSamples++;
			Instance tmpInst = trainInst.copy();
			tmpInst.setMissing(saIndex);
			double[] votes = learner.getVotesForInstance(tmpInst);
			evaluator.addResult(trainInstanceExample, votes);
			if (stopTrain==false)//training from beginning until bias occurs
				learner.trainOnInstance(trainInst);

			if (numberSamples>=windowSize){
				if (numberSamples%windowSize==0){
					cnt++;
					//for calculation of discrimination score
					String[] labels = evaluator.getAucEstimator().getSAVal();
					double[] predictions = evaluator.getAucEstimator().getPredictions();
					int[] trueLabels = evaluator.getAucEstimator().getTrueLabel();
					double discClassifier=DiscriminationScore(labels,predictions,trueLabels);
					avgDisc+=Math.abs(discClassifier);
					double discData = Disc_Data();
					if (Math.abs(discClassifier)>epsilon){ numOfChanges +=1;
						if (stopTrain==false){
							learner.resetLearning();
							stopTrain=true;
						}
						ApplyReweighing();
					}else{//no disc in the window -> train with new weight
						if (favPos!=0||favNeg!=0||savPos!=0||savNeg!=0){
							for (int i=0; i<windowSize; i++){
								String[] splits=windowList[i].instance.toString().split(",");
								int cl=Integer.parseInt(splits[splits.length-1]);
								if (splits[saIndex].equals(saVal)){//Deprived					
									if (cl==desiredClass)//Positive class
										windowList[i].instance.setWeight(savPos);
									else
										windowList[i].instance.setWeight(savNeg);
								}else{
									if (cl==desiredClass)//Positive class
										windowList[i].instance.setWeight(favPos);
									else
										windowList[i].instance.setWeight(favNeg);
								}
							}
						}
					}
					for (int i=0; i<windowSize; i++)
						learner.trainOnInstance(windowList[i]);
					//evaluation
					double precision= evaluator.getAucEstimator().getPrecision();
					double acc = evaluator.getAucEstimator().getAccuracy();
					double recall=evaluator.getAucEstimator().getRecall();
					double f1=0;
					if (precision!=0||recall!=0)
						f1=2*precision*recall/(precision+recall);
					double rocArea= evaluator.getAucEstimator().getAUC();
					double prArea=evaluator.getAucEstimator().prArea();
					avgAcc+=acc;
					avgF1+=f1;
					avgROC+=rocArea;
					avgPR+=prArea;
					double kappa=evaluator.getAucEstimator().getKappa();
					double kappaM=evaluator.getAucEstimator().getKappaM();
					double gMean=evaluator.getAucEstimator().getGMean();
					avgKappa+=kappa;
					avgKappaM+=kappaM;
					avgGMean+=gMean;
					writer.println(numberSamples+","+discData+","+discClassifier+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);

				}
			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numOfChanges);
		writer.close();
		System.out.println("Done 11.");
	}
	//model 12
	//ResetLearner		PredictNoSA:x	train_same_weights			train_new_weights
	public static void AccClassifierNoTrain_PredictNoSA(String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		windowList = new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false; int numOfChanges = 0;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;

			numberSamples++;
			Instance tmpInst = trainInst.copy();
			tmpInst.setMissing(saIndex);
			double[] votes = learner.getVotesForInstance(tmpInst);
			evaluator.addResult(trainInstanceExample, votes);
			if (stopTrain==false)//training from beginning until bias occurs
				learner.trainOnInstance(trainInst);

			if (numberSamples>=windowSize){
				if (numberSamples%windowSize==0){
					cnt++;
					//for calculation of discrimination score
					String[] labels = evaluator.getAucEstimator().getSAVal();
					double[] predictions = evaluator.getAucEstimator().getPredictions();
					int[] trueLabels = evaluator.getAucEstimator().getTrueLabel();
					double discClassifier=DiscriminationScore(labels,predictions,trueLabels);
					avgDisc+=Math.abs(discClassifier);
					double discData = Disc_Data();
					if (Math.abs(discClassifier)>epsilon){ numOfChanges +=1;
						if (stopTrain==false){
							stopTrain=true;
							learner.resetLearning();
						}
						ApplyReweighing();
						for (int i=0; i<windowSize; i++)
							learner.trainOnInstance(windowList[i]);
					}
					//evaluation
					double precision= evaluator.getAucEstimator().getPrecision();
					double acc = evaluator.getAucEstimator().getAccuracy();
					double recall=evaluator.getAucEstimator().getRecall();
					double f1=0;
					if (precision!=0||recall!=0)
						f1=2*precision*recall/(precision+recall);
					double rocArea= evaluator.getAucEstimator().getAUC();
					double prArea=evaluator.getAucEstimator().prArea();
					avgAcc+=acc;
					avgF1+=f1;
					avgROC+=rocArea;
					avgPR+=prArea;
					double kappa=evaluator.getAucEstimator().getKappa();
					double kappaM=evaluator.getAucEstimator().getKappaM();
					double gMean=evaluator.getAucEstimator().getGMean();
					avgKappa+=kappa;
					avgKappaM+=kappaM;
					avgGMean+=gMean;
					writer.println(numberSamples+","+discData+","+discClassifier+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);

				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numOfChanges);
		writer.close();
		System.out.println("Done 12.");
	}

	public static double DiscriminationScore(String[] labels, double[] predictions, int[] trueLabels){
		tpDeprived=0;tnDeprived=0;fnDeprived=0;fpDeprived=0;
		tpFavored=0;tnFavored=0;fnFavored=0;fpFavored=0;
		for (int i=0; i<windowSize; i++){
			if (labels[i].equals(saVal)){ //Deprived

//				if (predictions[i]==trueLabels[i]){//correctly predicted
				if (predictions[i]==1.0){//correctly predicted
					if (trueLabels[i]==desiredClass)//positive
						tpDeprived++;
					else
						tnDeprived++;
				}else{//incorrectly predicted
					if (trueLabels[i]==desiredClass)//positive => predict to => negative
						fnDeprived++;
					else
						fpDeprived++;
				}
			}else{//Favored

//				if (predictions[i]==trueLabels[i]){//correctly predicted
				if (predictions[i]==1.0){//correctly predicted
					if (trueLabels[i]==desiredClass)//positive
						tpFavored++;
					else
						tnFavored++;
				}else{//incorrectly predicted
					if (trueLabels[i]==desiredClass)//positive => predict to => negative
						fnFavored++;
					else
						fpFavored++;
				}
			}
		}
		saPos=tpDeprived+fnDeprived;
		saNeg=tnDeprived+fpDeprived;
		nSaPos=tpFavored+fnFavored;
		nSaNeg=tnFavored+fpFavored;
		System.out.println("saPos= " +saPos + " saNeg= " + saNeg + " nSaPos= " +nSaPos + " nSaNeg= "+ nSaNeg);
		if ((saPos+saNeg)==0)
			return 100*(double)(tpFavored+fpFavored)/(double)(nSaPos+nSaNeg);
		else{
			if ((nSaPos+nSaNeg)==0)
				return -(double)(tpDeprived+fpDeprived)/(double)(saPos+saNeg);
			else
				System.out.println("alles gut");
				System.out.println("true positive Male= " + tpFavored);
				System.out.println("true negative Male= " + tnFavored);
				System.out.println("false positive Male= " + fpFavored);
				System.out.println("false negative Male= " + fnFavored);

//				System.out.println("tpFavored= " + tpFavored);
				System.out.println("male percentage= " + (double)(tpFavored+fpFavored)/(double)(nSaPos+nSaNeg));
				System.out.println("female percentage= " + (double)(tpDeprived+fpDeprived)/(double)(saPos+saNeg));

				return 100*((double)(tpFavored+fpFavored)/(double)(nSaPos+nSaNeg)
						-(double)(tpDeprived+fpDeprived)/(double)(saPos+saNeg));
		}
	}
	public static double Disc_Data(){//DiscriminationScore should be calculated first
		saPos=tpDeprived+fnDeprived;
		saNeg=tnDeprived+fpDeprived;
		nSaPos=tpFavored+fnFavored;
		nSaNeg=tnFavored+fpFavored;
		if ((saPos+saNeg)==0)
			return 100*(double)nSaPos/(double)(nSaPos+nSaNeg);
		else{
			if ((nSaPos+nSaNeg)==0)
				return -(double)saPos/(double)(saPos+saNeg);
			else
				return 100*((double)(nSaPos)/(double)(nSaPos+nSaNeg)
						-(double)(saPos)/(double)(saPos+saNeg));
		}
	}
	public static void ApplyReweighing(){
		//weight calculation
		savPos=0;savNeg=0;favPos=0;favNeg=0;
		if (saPos!=0)
			savPos=(double)(saPos+saNeg)*(double)(saPos+nSaPos)/(double)(windowSize*saPos);
		else
			savPos=1;
		if (saNeg!=0)
			savNeg=(double)(saPos+saNeg)*(double)(saNeg+nSaNeg)/(double)(windowSize*saNeg);
		else
			savNeg=1;
		if (nSaPos!=0)
			favPos=(double)(nSaPos+nSaNeg)*(double)(saPos+nSaPos)/(double)(windowSize*nSaPos);
		else
			favPos=1;
		if (nSaNeg!=0)
			favNeg=(double)(nSaPos+nSaNeg)*(double)(saNeg+nSaNeg)/(double)(windowSize*nSaNeg);
		else
			favNeg=1;
		//apply new weight for the current window
		for (int i=0; i<windowSize; i++){
			String[] splits=windowList[i].instance.toString().split(",");
			int cl=Integer.parseInt(splits[splits.length-1]);
			if (splits[saIndex].equals(saVal)){//Deprived					
				if (cl==desiredClass)//Positive class
					windowList[i].instance.setWeight(savPos);
				else
					windowList[i].instance.setWeight(savNeg);
			}else{
				if (cl==desiredClass)//Positive class
					windowList[i].instance.setWeight(favPos);
				else
					windowList[i].instance.setWeight(favNeg);
			}
		}
	}
}
