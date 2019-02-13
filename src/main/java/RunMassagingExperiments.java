import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.Classifier;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.lazy.kNN;
import moa.classifiers.meta.AccuracyUpdatedEnsemble;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.InstanceExample;
import moa.streams.ArffFileStream;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;

import static java.lang.System.exit;

public class RunMassagingExperiments {
	protected static int saPos = 0, saNeg=0, nSaPos = 0, nSaNeg=0;
	protected static int tpDeprived=0, fpDeprived=0, tpFavored=0, fpFavored=0;
	protected static int tnDeprived=0, fnDeprived=0, tnFavored=0, fnFavored=0;

	protected static int windowSize;
	protected static String saVal ;
	protected static String saName ;
	protected static int epsilon; //percentage
	//data definition
	protected static int saIndex;
	protected static int desiredClass ;
	protected static int notDesiredClass ;

	protected static InstanceExample[] windowList = new InstanceExample[windowSize];

	protected static double[][] sortedPromotionList;
	protected static double[][] sortedDemotionList;

	//different classifiers
	protected static Classifier ranker = new NaiveBayes();
	protected static Classifier learner  ;
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
//		saIndex=0;
//		desiredClass = 1;
//		notDesiredClass = 0;
//		saName = "sex";
//		saVal = "Female";

		String outfile ="";
		String infile ="";
		String infileWithoutSA="";

		if(args.length != 4){
			System.out.println("1:B/S/G 2:epsilon_int 3:window_size_int 4:NB/HT/KNN/AUE");
			exit(1);
		}
		epsilon= Integer.valueOf(args[1]);
		windowSize 	= Integer.valueOf(args[2]);
//		epsilon= 0;
//		windowSize 	= 1000;
//
//        saIndex=2;
//        desiredClass = 1;
//        notDesiredClass = 0;
//        saName = "SA";
//        saVal = "Female";
////
//        infile = "/home/iosifidis/HansCode/synthetic_data.arff";
//        infileWithoutSA = "/home/iosifidis/HansCode/synthetic_data_NOSA.arff";
//        outfile ="/home/iosifidis/HansCode/TIME_RESULTS/StreamGenerator/epsilon_" + epsilon + "/Classifier_test/window_" + windowSize + "/";
//        learner = new AccuracyUpdatedEnsemble();//5.


		if(args[0].equals("B")) {
			infile = "/home/iosifidis/HansCode/big_remove_InstanceWeight_shuffling1.arff";
			infileWithoutSA = "/home/iosifidis/HansCode/big_remove_InstanceWeight_shuffling1_SA.arff";
			outfile = "/home/iosifidis/HansCode/TIME_RESULTS/BigDataset/epsilon_" + epsilon + "/Classifier_" + args[3] +"/window_" + windowSize + "/";
		}else if(args[0].equals("S")){
			infile="/home/iosifidis/HansCode/small_dataset_remove_fnlwgt_shuffling1.arff";
			infileWithoutSA = "/home/iosifidis/HansCode/small_dataset_remove_fnlwgt_shuffling1_SA.arff";
			outfile = "/home/iosifidis/HansCode/TIME_RESULTS/SmallDataset/epsilon_" + epsilon + "/Classifier_" + args[3]+ "/window_" + windowSize + "/";
		} else if (args[0].equals("G")){
			saIndex=2;
			desiredClass = 1;
			notDesiredClass = 0;
			saName = "SA";
			saVal = "Female";


			infile = "/home/iosifidis/HansCode/synthetic_data.arff";
			infileWithoutSA = "/home/iosifidis/HansCode/synthetic_data_NOSA.arff";
//			outfile ="/home/iosifidis/HansCode/TIME_RESULTS/StreamGenerator/epsilon_" + epsilon + "/Classifier_test/window_" + windowSize + "/";

			outfile ="/home/iosifidis/HansCode/TIME_RESULTS/StreamGenerator/epsilon_" + epsilon + "/Classifier_" + args[3]+ "/window_" + windowSize + "/";
		} else {
			exit(1);
		}

		new File(outfile).mkdirs();

		if(args[3].equals("NB")){
			learner = new NaiveBayes();//5.
		} else if (args[3].equals("HT")){
			learner = new HoeffdingTree();//5.
		} else if (args[3].equals("KNN")){
			learner = new kNN();//5.
		} else if (args[3].equals("AUE")){
			learner = new AccuracyUpdatedEnsemble();//5.
		} else{
			exit(1);
		}

		PrintWriter writer = new PrintWriter(outfile + "time.csv");

//
		long startTime = System.currentTimeMillis();
		NoResetLearnerResetRanker(infileWithoutSA, infile, outfile + "AccumWithSA"+".csv");//1.
		long endTime = System.currentTimeMillis();
		writer.println("AccumWithSA:" + (endTime - startTime));

		ranker.resetLearning();
		learner.resetLearning();

		startTime = System.currentTimeMillis();
		ResetLearnerResetRanker(infileWithoutSA, infile, outfile+"ResetWithSA"+".csv");//2.
		endTime = System.currentTimeMillis();
		writer.println("ResetWithSA:" + (endTime - startTime));
//
		ranker.resetLearning();
		learner.resetLearning();

		startTime = System.currentTimeMillis();
		NoResetLearnerResetRanker_PredictNoSA(infileWithoutSA, infile, outfile+"AccumNoSA"+".csv");//5.
		endTime = System.currentTimeMillis();
		writer.println("AccumNoSA:" + (endTime - startTime));

		ranker.resetLearning();
		learner.resetLearning();

		startTime = System.currentTimeMillis();
		ResetLearnerResetRanker_PredictNoSA(infileWithoutSA,infile,outfile+"ResetNoSA"+".csv");//6.
		endTime = System.currentTimeMillis();
		writer.println("ResetNoSA:" + (endTime - startTime));

		ranker.resetLearning();
		learner.resetLearning();

		startTime = System.currentTimeMillis();
		NoResetLearnerResetRanker_LearnerNoContLearn(infileWithoutSA,infile,outfile+"AccumNoTrainWithSA"+".csv");//9.
		endTime = System.currentTimeMillis();
		writer.println("AccumNoTrainWithSA:" + (endTime - startTime));
//
		ranker.resetLearning();
		learner.resetLearning();
//
		startTime = System.currentTimeMillis();
		ResetLearnerResetRanker_LearnerNoContLearn(infileWithoutSA,infile,outfile+"ResetNoTrainWithSA"+".csv");//10.
		endTime = System.currentTimeMillis();
		writer.println("ResetNoTrainWithSA:" + (endTime - startTime));

		ranker.resetLearning();
		learner.resetLearning();

		startTime = System.currentTimeMillis();
		NoResetLearnerResetRanker_PredictNoSA_LearnerNoContLearn(infileWithoutSA,infile,outfile+"AccumNoTrainNoSA"+".csv");//13.
		endTime = System.currentTimeMillis();
		writer.println("AccumNoTrainNoSA:" + (endTime - startTime));

		ranker.resetLearning();
		learner.resetLearning();

		startTime = System.currentTimeMillis();
		ResetLearnerResetRanker_PredictNoSA_LearnerNoContLearn(infileWithoutSA,infile,outfile+"ResetNoTrainNoSA"+".csv");//14.
		endTime = System.currentTimeMillis();
		writer.println("ResetNoTrainNoSA:" + (endTime - startTime));

		ranker.resetLearning();
		learner.resetLearning();


		startTime = System.currentTimeMillis();
		KeepReseting(infile,outfile+"Baseline_reset"  + ".csv");
		endTime = System.currentTimeMillis();
		writer.println("Baseline_reset:" + (endTime - startTime));

		ranker.resetLearning();
		learner.resetLearning();

		startTime = System.currentTimeMillis();
		KeepResetingWithoutSA(infile, outfile + "Baseline_reset_no_SA.csv");
		endTime = System.currentTimeMillis();
		writer.println("Baseline_reset_no_SA:" + (endTime - startTime));

		ranker.resetLearning();
		learner.resetLearning();

//		startTime = System.currentTimeMillis();
//		Massaging1stWindow(infile, outfile + "Baseline_massage"   + ".csv");
//		endTime = System.currentTimeMillis();
//		writer.println("Baseline_massage:" + (endTime - startTime));
//
//		ranker.resetLearning();
//		learner.resetLearning();

		startTime = System.currentTimeMillis();
		batchRemoveSA(infileWithoutSA, infile, outfile + "Baseline_NoSA" + ".csv");
		endTime = System.currentTimeMillis();
		writer.println("Baseline_NoSA:" + (endTime - startTime));
		writer.close();
		ranker.resetLearning();
		learner.resetLearning();
	}




	//baseline: keep reseting without SA
	public static void KeepResetingWithoutSA(String filename, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(filename, -1);
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

		int numberSamples=0;int cnt=0; int numOfChanges = 0;
		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,changes, Acc, P, R, F1, rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		windowList = new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			numberSamples++;
			//evaluator window

			Instance tmpInst = trainInst.copy();
			tmpInst.setMissing(saIndex);
			double[] votes = learner.getVotesForInstance(tmpInst);

			evaluator.addResult(trainInstanceExample, votes);
			learner.trainOnInstance(trainInst);
			if (numberSamples>=windowSize){
				if (numberSamples%windowSize==0){
					cnt++;
					String[] labels;
					double[] predictions;
					int[] trueLabels;
					labels = evaluator.getAucEstimator().getSAVal();
					predictions = evaluator.getAucEstimator().getPredictions();
					trueLabels = evaluator.getAucEstimator().getTrueLabel();
					double discClassifier=DiscriminationScore(labels,predictions,trueLabels);
					avgDisc+=Math.abs(discClassifier);
					double discData = Disc_Data();

					if (Math.abs(discClassifier)>epsilon){
						numOfChanges +=1;
						learner.resetLearning();
						for (int i=0; i<windowSize; i++){
							learner.trainOnInstance(windowList[i]);
						}
					}

					//evaluation
					double precision= evaluator.getAucEstimator().getPrecision();
					double acc = evaluator.getAucEstimator().getAccuracy();
					avgAcc+=acc;
					double recall=evaluator.getAucEstimator().getRecall();
					double f1=0;
					if (precision!=0||recall!=0)
						f1=2*precision*recall/(precision+recall);
					avgF1+=f1;
					double rocArea= evaluator.getAucEstimator().getAUC();
					avgROC+=rocArea;
					double prArea=evaluator.getAucEstimator().prArea();
					avgPR+=prArea;
					double kappa=evaluator.getAucEstimator().getKappa();
					avgKappa+=kappa;
					double kappaM=evaluator.getAucEstimator().getKappaM();
					avgKappaM+=kappaM;
					double gMean=evaluator.getAucEstimator().getGMean();
					avgGMean+=gMean;
					writer.println(numberSamples+","+discData+","+discClassifier+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);
				}
			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt
				+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","
				+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numOfChanges);
		writer.close();
		System.out.println("<WITHOUT_SA> DONE BASELINE KEEP RESETING WHEN GETTING BIAS");
	}


	//greedyModel: learns Everything
	public static void greedyModel(String infileWithoutSA,String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		ArffFileStream fsWithoutSA = new ArffFileStream(infileWithoutSA, -1);
		fsWithoutSA.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		ranker.setModelContext(fsWithoutSA.getHeader());
		ranker.prepareForUse();
		WindowAUCImbalancedPerformanceEvaluator ranker_evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		ranker_evaluator.widthOption.setValue(windowSize);
		ranker_evaluator.setIndex(saIndex);
		ranker_evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,changes,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		double changes=0; int numberOfchanges = 0;
		windowList = new InstanceExample[windowSize];
		InstanceExample[] windowForRanker=new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			InstanceExample InstForRanker = fsWithoutSA.nextInstance();
			windowForRanker[numberSamples % windowSize]=InstForRanker;
			numberSamples++;
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);
			//training from beginning until bias occurs
//			if (stopTrain==false)
			learner.trainOnInstance(trainInst);

			ranker.trainOnInstance(InstForRanker);

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
					changes=0;
					if (discClassifier>epsilon){
						numberOfchanges +=1;
						//massaging
						int saNum = saPos+saNeg;
						int nSaNum= nSaPos+nSaNeg;
						if (discClassifier<-epsilon){//reverse discrimination & previous did the change							
							changes=((double)saPos*(double)nSaNum-(double)nSaPos*(double)saNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
							if (changes>0){
								if (saVal.equals("Female"))
									saVal="Male";
								else
									saVal="Female";
							}
						}else{
							changes=((double)nSaPos*(double)saNum-(double)saPos*(double)nSaNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
						}

						if (changes>0){ //massaging taking place
							if (stopTrain==false){
//								learner.resetLearning();
								stopTrain=true;
							}
							//predict on the current window
							for (int i=0; i<windowSize; i++){
								double[] ranker_votes = ranker.getVotesForInstance(windowForRanker[i]);
								ranker_evaluator.addResult(windowForRanker[i], ranker_votes);
							}
							int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
							int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();
//							String[] saValFromSortedScores=ranker_evaluator.getAucEstimator().getSAValFromSortedScores();
							double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
							String[] saValFromSortedScores=new String[windowSize];
							for (int i=0; i<windowSize; i++){
								String[] splits=windowList[posWindow[i]%windowSize].instance.toString().split(",");
								saValFromSortedScores[i]=splits[saIndex];
							}
							rankingWithSA(posWindow,saValFromSortedScores,sortedLabels,sortedScores);
							relabel_M(changes);
							for (int i=0; i<windowSize; i++){
								learner.trainOnInstance(windowList[i]);
							}
						}
//						if (stopTrain==true){
//							for (int i=0; i<windowSize; i++){
//								learner.trainOnInstance(windowList[i]);
//							}
//						}
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
					writer.println(numberSamples+","+discData+","+discClassifier+","+changes+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);
				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,Changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numberOfchanges);
		writer.close();
		System.out.println("Done GreedyModel ");
	}
	//model 1:Learner NoReset, Ranker learns from each window (withoutSA)
	//RankerNoSA:Y	ResetRanker:Y	ResetLearner	PredictNoSA
	public static void NoResetLearnerResetRanker(String infileWithoutSA,String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}

		fs.prepareForUse();

		ArffFileStream fsWithoutSA = new ArffFileStream(infileWithoutSA, -1);
		fsWithoutSA.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();
		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,changes,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		double changes=0; int numberOfchanges = 0;
		windowList = new InstanceExample[windowSize];
		InstanceExample[] windowForRanker=new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			windowForRanker[numberSamples%windowSize]=fsWithoutSA.nextInstance();

			numberSamples++;
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);
			//training from beginning until bias occurs
			if (stopTrain==false)
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
 					changes=0;
					if (Math.abs(discClassifier)>epsilon){
//						System.out.println("discClassifier = " + discClassifier);
//						System.out.println("numberSamples = " + numberSamples);
						numberOfchanges += 1;
						//massaging
						int saNum = saPos+saNeg;
						int nSaNum= nSaPos+nSaNeg;
						if (discClassifier<-epsilon){//reverse discrimination & previous did the change							
							changes=((double)saPos*(double)nSaNum-(double)nSaPos*(double)saNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
							if (changes>0){
								if (saVal.equals("Female"))
									saVal="Male";
								else
									saVal="Female";
							}
						}else{
							changes=((double)nSaPos*(double)saNum-(double)saPos*(double)nSaNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
						}

						if (changes>0){ //massaging taking place
							if (stopTrain==false){
								learner.resetLearning();
								stopTrain=true;
							}
							//ranker
							ranker = new NaiveBayes();
							ranker.setModelContext(fsWithoutSA.getHeader());
							ranker.prepareForUse();
							WindowAUCImbalancedPerformanceEvaluator ranker_evaluator =new WindowAUCImbalancedPerformanceEvaluator();
							ranker_evaluator.widthOption.setValue(windowSize);
							ranker_evaluator.setIndex(saIndex);
							ranker_evaluator.prepareForUse();

							for (int i=0; i<windowSize; i++){
								ranker.trainOnInstance(windowForRanker[i]);
							}
							for (int i=0; i<windowSize; i++){
								double[] ranker_votes = ranker.getVotesForInstance(windowForRanker[i]);
								ranker_evaluator.addResult(windowForRanker[i], ranker_votes);
							}
							int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
							int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();
//							String[] saValFromSortedScores=ranker_evaluator.getAucEstimator().getSAValFromSortedScores();
							double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
							String[] saValFromSortedScores=new String[windowSize];
							for (int i=0; i<windowSize; i++){
								String[] splits=windowList[posWindow[i]%windowSize].instance.toString().split(",");
								saValFromSortedScores[i]=splits[saIndex];
							}
							rankingWithSA(posWindow,saValFromSortedScores,sortedLabels,sortedScores);
							relabel_M(changes);
						}

					}if (stopTrain==true){
						for (int i=0; i<windowSize; i++){
							learner.trainOnInstance(windowList[i]);
						}
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
					writer.println(numberSamples+","+discData+","+discClassifier+","+changes+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);
				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println((avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numberOfchanges));
		writer.close();
		System.out.println("Done 1.");
	}
	//model 2: RankerNoSA:Y	ResetRanker:Y	ResetLearner:Y	PredictNoSA
	public static void ResetLearnerResetRanker(String infileWithoutSA,String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		ArffFileStream fsWithoutSA = new ArffFileStream(infileWithoutSA, -1);
		fsWithoutSA.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,changes,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		double changes=0; int numberOfchanges = 0;
		windowList = new InstanceExample[windowSize];
		InstanceExample[] windowForRanker=new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			windowForRanker[numberSamples%windowSize]=fsWithoutSA.nextInstance();

			numberSamples++;
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);
			//training from beginning until bias occurs
			if (stopTrain==false)
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
					changes=0;
					if (Math.abs(discClassifier)>epsilon){
						numberOfchanges +=1;
						//massaging
						int saNum = saPos+saNeg;
						int nSaNum= nSaPos+nSaNeg;
						if (discClassifier<-epsilon){//reverse discrimination & previous did the change							
							changes=((double)saPos*(double)nSaNum-(double)nSaPos*(double)saNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
							if (changes>0){
								if (saVal.equals("Female"))
									saVal="Male";
								else
									saVal="Female";
							}
						}else{
							changes=((double)nSaPos*(double)saNum-(double)saPos*(double)nSaNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
						}

						if (changes>0){ //massaging taking place
							learner.resetLearning();
							stopTrain=true;

							//ranker
							ranker = new NaiveBayes();
							ranker.setModelContext(fsWithoutSA.getHeader());
							ranker.prepareForUse();
							WindowAUCImbalancedPerformanceEvaluator ranker_evaluator =new WindowAUCImbalancedPerformanceEvaluator();
							ranker_evaluator.widthOption.setValue(windowSize);
							ranker_evaluator.setIndex(saIndex);
							ranker_evaluator.prepareForUse();

							for (int i=0; i<windowSize; i++){
								ranker.trainOnInstance(windowForRanker[i]);
							}
							for (int i=0; i<windowSize; i++){
								double[] ranker_votes = ranker.getVotesForInstance(windowForRanker[i]);
								ranker_evaluator.addResult(windowForRanker[i], ranker_votes);
							}
							int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
							int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();
//							String[] saValFromSortedScores=ranker_evaluator.getAucEstimator().getSAValFromSortedScores();
							double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
							String[] saValFromSortedScores=new String[windowSize];
							for (int i=0; i<windowSize; i++){
								String[] splits=windowList[posWindow[i]%windowSize].instance.toString().split(",");
								saValFromSortedScores[i]=splits[saIndex];
							}
							rankingWithSA(posWindow,saValFromSortedScores,sortedLabels,sortedScores);
							relabel_M(changes);
						}

					}if (stopTrain==true){
						for (int i=0; i<windowSize; i++){
							learner.trainOnInstance(windowList[i]);
						}
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
					writer.println(numberSamples+","+discData+","+discClassifier+","+changes+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);
				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numberOfchanges);
		writer.close();
		System.out.println("Done 2. ");
	}
	//model 3: RankerNoSA:Y	ResetRanker		ResetLearner:Y	PredictNoSA
	public static void ResetLearnerNoResetRanker(String infileWithoutSA,String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		ArffFileStream fsWithoutSA = new ArffFileStream(infileWithoutSA, -1);
		fsWithoutSA.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		ranker.setModelContext(fsWithoutSA.getHeader());
		ranker.prepareForUse();
		WindowAUCImbalancedPerformanceEvaluator ranker_evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		ranker_evaluator.widthOption.setValue(windowSize);
		ranker_evaluator.setIndex(saIndex);
		ranker_evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,changes,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		double changes=0; int numberOfchanges = 0;
		windowList = new InstanceExample[windowSize];
		InstanceExample[] windowForRanker=new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			InstanceExample InstForRanker = fsWithoutSA.nextInstance();
			windowForRanker[numberSamples % windowSize]=InstForRanker;
			numberSamples++;
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);
			//training from beginning until bias occurs
			if (stopTrain==false)
				learner.trainOnInstance(trainInst);

			ranker.trainOnInstance(InstForRanker);

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
					changes=0;
					if (Math.abs(discClassifier)>epsilon){
						numberOfchanges +=1;
						//massaging
						int saNum = saPos+saNeg;
						int nSaNum= nSaPos+nSaNeg;
						if (discClassifier<-epsilon){//reverse discrimination & previous did the change							
							changes=((double)saPos*(double)nSaNum-(double)nSaPos*(double)saNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
							if (changes>0){
								if (saVal.equals("Female"))
									saVal="Male";
								else
									saVal="Female";
							}
						}else{
							changes=((double)nSaPos*(double)saNum-(double)saPos*(double)nSaNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
						}

						if (changes>0){ //massaging taking place
							learner.resetLearning();
							stopTrain=true;
							//predict on the current window
							for (int i=0; i<windowSize; i++){
								double[] ranker_votes = ranker.getVotesForInstance(windowForRanker[i]);
								ranker_evaluator.addResult(windowForRanker[i], ranker_votes);
							}
							int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
							int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();
//							String[] saValFromSortedScores=ranker_evaluator.getAucEstimator().getSAValFromSortedScores();
							double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
							String[] saValFromSortedScores=new String[windowSize];
							for (int i=0; i<windowSize; i++){
								String[] splits=windowList[posWindow[i]%windowSize].instance.toString().split(",");
								saValFromSortedScores[i]=splits[saIndex];
							}
							rankingWithSA(posWindow,saValFromSortedScores,sortedLabels,sortedScores);
							relabel_M(changes);
						}
						if (stopTrain==true){
							for (int i=0; i<windowSize; i++){
								learner.trainOnInstance(windowList[i]);
							}
						}
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
					writer.println(numberSamples+","+discData+","+discClassifier+","+changes+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);
				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numberOfchanges);
		writer.close();
		System.out.println("Done 3. ");
	}
	//model 4: RankerNoSA:Y	ResetRanker		ResetLearner	PredictNoSA
	public static void NoResetLearnerNoResetRanker(String infileWithoutSA,String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		ArffFileStream fsWithoutSA = new ArffFileStream(infileWithoutSA, -1);
		fsWithoutSA.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		ranker.setModelContext(fsWithoutSA.getHeader());
		ranker.prepareForUse();
		WindowAUCImbalancedPerformanceEvaluator ranker_evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		ranker_evaluator.widthOption.setValue(windowSize);
		ranker_evaluator.setIndex(saIndex);
		ranker_evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,changes,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		double changes=0; int numberOfchanges = 0;
		windowList = new InstanceExample[windowSize];
		InstanceExample[] windowForRanker=new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			InstanceExample InstForRanker = fsWithoutSA.nextInstance();
			windowForRanker[numberSamples % windowSize]=InstForRanker;
			numberSamples++;
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);
			//training from beginning until bias occurs
			if (stopTrain==false)
				learner.trainOnInstance(trainInst);

			ranker.trainOnInstance(InstForRanker);

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
					changes=0;
					if (Math.abs(discClassifier)>epsilon){numberOfchanges += 1;
						//massaging
						int saNum = saPos+saNeg;
						int nSaNum= nSaPos+nSaNeg;
						if (discClassifier<-epsilon){//reverse discrimination & previous did the change							
							changes=((double)saPos*(double)nSaNum-(double)nSaPos*(double)saNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
							if (changes>0){
								if (saVal.equals("Female"))
									saVal="Male";
								else
									saVal="Female";
							}
						}else{
							changes=((double)nSaPos*(double)saNum-(double)saPos*(double)nSaNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
						}

						if (changes>0){ //massaging taking place
							if (stopTrain==false){
								learner.resetLearning();
								stopTrain=true;
							}
							//predict on the current window
							for (int i=0; i<windowSize; i++){
								double[] ranker_votes = ranker.getVotesForInstance(windowForRanker[i]);
								ranker_evaluator.addResult(windowForRanker[i], ranker_votes);
							}
							int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
							int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();
//							String[] saValFromSortedScores=ranker_evaluator.getAucEstimator().getSAValFromSortedScores();
							double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
							String[] saValFromSortedScores=new String[windowSize];
							for (int i=0; i<windowSize; i++){
								String[] splits=windowList[posWindow[i]%windowSize].instance.toString().split(",");
								saValFromSortedScores[i]=splits[saIndex];
							}
							rankingWithSA(posWindow,saValFromSortedScores,sortedLabels,sortedScores);
							relabel_M(changes);
						}
						if (stopTrain==true){
							for (int i=0; i<windowSize; i++){
								learner.trainOnInstance(windowList[i]);
							}
						}
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
					writer.println(numberSamples+","+discData+","+discClassifier+","+changes+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);
				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numberOfchanges);
		writer.close();
		System.out.println("Done 4. ");
	}
	//model 4'
	public static void NoResetLearnerNoResetRanker2(String infileWithoutSA,String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		ArffFileStream fsWithoutSA = new ArffFileStream(infileWithoutSA, -1);
		fsWithoutSA.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		ranker.setModelContext(fsWithoutSA.getHeader());
		ranker.prepareForUse();
		WindowAUCImbalancedPerformanceEvaluator ranker_evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		ranker_evaluator.widthOption.setValue(windowSize);
		ranker_evaluator.setIndex(saIndex);
		ranker_evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,changes,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		double changes=0; int numberOfchanges = 0;
		windowList = new InstanceExample[windowSize];
		InstanceExample[] windowForRanker=new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			InstanceExample InstForRanker = fsWithoutSA.nextInstance();
			windowForRanker[numberSamples % windowSize]=InstForRanker;
			numberSamples++;
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);
			//training from beginning until bias occurs
			if (stopTrain==false)
				learner.trainOnInstance(trainInst);

			ranker.trainOnInstance(InstForRanker);

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
					changes=0;
					if (Math.abs(discClassifier)>epsilon){numberOfchanges += 1;
						//massaging
						int saNum = saPos+saNeg;
						int nSaNum= nSaPos+nSaNeg;
						if (discClassifier<-epsilon){//reverse discrimination & previous did the change							
							changes=((double)saPos*(double)nSaNum-(double)nSaPos*(double)saNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
							if (changes>0){
								if (saVal.equals("Female"))
									saVal="Male";
								else
									saVal="Female";
							}
						}else{
							changes=((double)nSaPos*(double)saNum-(double)saPos*(double)nSaNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
						}

						if (changes>0){ //massaging taking place
							if (stopTrain==false){
//								learner.resetLearning();
								stopTrain=true;
							}
							//predict on the current window
							for (int i=0; i<windowSize; i++){
								double[] ranker_votes = ranker.getVotesForInstance(windowForRanker[i]);
								ranker_evaluator.addResult(windowForRanker[i], ranker_votes);
							}
							int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
							int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();
//							String[] saValFromSortedScores=ranker_evaluator.getAucEstimator().getSAValFromSortedScores();
							double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
							String[] saValFromSortedScores=new String[windowSize];
							for (int i=0; i<windowSize; i++){
								String[] splits=windowList[posWindow[i]%windowSize].instance.toString().split(",");
								saValFromSortedScores[i]=splits[saIndex];
							}
							rankingWithSA(posWindow,saValFromSortedScores,sortedLabels,sortedScores);
							relabel_M(changes);
						}
						if (stopTrain==true){
							for (int i=0; i<windowSize; i++){
								learner.trainOnInstance(windowList[i]);
							}
						}
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
					writer.println(numberSamples+","+discData+","+discClassifier+","+changes+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);
				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numberOfchanges);
		writer.close();
		System.out.println("Done 4. ");
	}
	//model 5: RankerNoSA:Y	ResetRanker:Y	ResetLearner	PredictNoSA:Y
	public static void NoResetLearnerResetRanker_PredictNoSA(String infileWithoutSA,String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		ArffFileStream fsWithoutSA = new ArffFileStream(infileWithoutSA, -1);
		fsWithoutSA.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,changes,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		double changes=0; int numberOfchanges = 0;
		windowList = new InstanceExample[windowSize];
		InstanceExample[] windowForRanker=new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			windowForRanker[numberSamples%windowSize]=fsWithoutSA.nextInstance();

			numberSamples++;
			Instance tmpInst = trainInst.copy();
			tmpInst.setMissing(saIndex);
			double[] votes = learner.getVotesForInstance(tmpInst);
			evaluator.addResult(trainInstanceExample, votes);
			//training from beginning until bias occurs
			if (stopTrain==false)
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
					changes=0;
					if (Math.abs(discClassifier)>epsilon){numberOfchanges += 1;
						//massaging
						int saNum = saPos+saNeg;
						int nSaNum= nSaPos+nSaNeg;
						if (discClassifier<-epsilon){//reverse discrimination & previous did the change							
							changes=((double)saPos*(double)nSaNum-(double)nSaPos*(double)saNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
							if (changes>0){
								if (saVal.equals("Female"))
									saVal="Male";
								else
									saVal="Female";
							}
						}else{
							changes=((double)nSaPos*(double)saNum-(double)saPos*(double)nSaNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
						}

						if (changes>0){ //massaging taking place
							if (stopTrain==false){
								learner.resetLearning();
								stopTrain=true;
							}
							//ranker
							ranker = new NaiveBayes();
							ranker.setModelContext(fsWithoutSA.getHeader());
							ranker.prepareForUse();
							WindowAUCImbalancedPerformanceEvaluator ranker_evaluator =new WindowAUCImbalancedPerformanceEvaluator();
							ranker_evaluator.widthOption.setValue(windowSize);
							ranker_evaluator.setIndex(saIndex);
							ranker_evaluator.prepareForUse();

							for (int i=0; i<windowSize; i++){
								ranker.trainOnInstance(windowForRanker[i]);
							}
							for (int i=0; i<windowSize; i++){
								double[] ranker_votes = ranker.getVotesForInstance(windowForRanker[i]);
								ranker_evaluator.addResult(windowForRanker[i], ranker_votes);
							}
							int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
							int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();
//							String[] saValFromSortedScores=ranker_evaluator.getAucEstimator().getSAValFromSortedScores();
							double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
							String[] saValFromSortedScores=new String[windowSize];
							for (int i=0; i<windowSize; i++){
								String[] splits=windowList[posWindow[i]%windowSize].instance.toString().split(",");
								saValFromSortedScores[i]=splits[saIndex];
							}
							rankingWithSA(posWindow,saValFromSortedScores,sortedLabels,sortedScores);
							relabel_M(changes);
						}

					}if (stopTrain==true){
						for (int i=0; i<windowSize; i++){
							learner.trainOnInstance(windowList[i]);
						}
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
					writer.println(numberSamples+","+discData+","+discClassifier+","+changes+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);
				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numberOfchanges);
		writer.close();
		System.out.println("Done 5.");
	}
	//model 6: RankerNoSA:Y	ResetRanker:Y	ResetLearner:Y	PredictNoSA:Y
	public static void ResetLearnerResetRanker_PredictNoSA(String infileWithoutSA,String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		ArffFileStream fsWithoutSA = new ArffFileStream(infileWithoutSA, -1);
		fsWithoutSA.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,changes,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		double changes=0; int numberOfchanges = 0;
		windowList = new InstanceExample[windowSize];
		InstanceExample[] windowForRanker=new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			windowForRanker[numberSamples%windowSize]=fsWithoutSA.nextInstance();

			numberSamples++;
			Instance tmpInst = trainInst.copy();
			tmpInst.setMissing(saIndex);
			double[] votes = learner.getVotesForInstance(tmpInst);
			evaluator.addResult(trainInstanceExample, votes);
			//training from beginning until bias occurs
			if (stopTrain==false)
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
					changes=0;
					if (Math.abs(discClassifier)>epsilon){numberOfchanges += 1;
						//massaging
						int saNum = saPos+saNeg;
						int nSaNum= nSaPos+nSaNeg;
						if (discClassifier<-epsilon){//reverse discrimination & previous did the change							
							changes=((double)saPos*(double)nSaNum-(double)nSaPos*(double)saNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
							if (changes>0){
								if (saVal.equals("Female"))
									saVal="Male";
								else
									saVal="Female";
							}
						}else{
							changes=((double)nSaPos*(double)saNum-(double)saPos*(double)nSaNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
						}

						if (changes>0){ //massaging taking place
							learner.resetLearning();
							stopTrain=true;

							//ranker
							ranker = new NaiveBayes();
							ranker.setModelContext(fsWithoutSA.getHeader());
							ranker.prepareForUse();
							WindowAUCImbalancedPerformanceEvaluator ranker_evaluator =new WindowAUCImbalancedPerformanceEvaluator();
							ranker_evaluator.widthOption.setValue(windowSize);
							ranker_evaluator.setIndex(saIndex);
							ranker_evaluator.prepareForUse();

							for (int i=0; i<windowSize; i++){
								ranker.trainOnInstance(windowForRanker[i]);
							}
							for (int i=0; i<windowSize; i++){
								double[] ranker_votes = ranker.getVotesForInstance(windowForRanker[i]);
								ranker_evaluator.addResult(windowForRanker[i], ranker_votes);
							}
							int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
							int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();
//							String[] saValFromSortedScores=ranker_evaluator.getAucEstimator().getSAValFromSortedScores();
							double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
							String[] saValFromSortedScores=new String[windowSize];
							for (int i=0; i<windowSize; i++){
								String[] splits=windowList[posWindow[i]%windowSize].instance.toString().split(",");
								saValFromSortedScores[i]=splits[saIndex];
							}
							rankingWithSA(posWindow,saValFromSortedScores,sortedLabels,sortedScores);
							relabel_M(changes);
						}

					}if (stopTrain==true){
						for (int i=0; i<windowSize; i++){
							learner.trainOnInstance(windowList[i]);
						}
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
					writer.println(numberSamples+","+discData+","+discClassifier+","+changes+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);
				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numberOfchanges);
		writer.close();
		System.out.println("Done 6. ");
	}
	//model 7: RankerNoSA:Y	ResetRanker		ResetLearner:Y	PredictNoSA:Y
	public static void ResetLearnerNoResetRanker_PredictNoSA(String infileWithoutSA,String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		ArffFileStream fsWithoutSA = new ArffFileStream(infileWithoutSA, -1);
		fsWithoutSA.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		ranker.setModelContext(fsWithoutSA.getHeader());
		ranker.prepareForUse();
		WindowAUCImbalancedPerformanceEvaluator ranker_evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		ranker_evaluator.widthOption.setValue(windowSize);
		ranker_evaluator.setIndex(saIndex);
		ranker_evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,changes,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		double changes=0; int numberOfchanges = 0;
		windowList = new InstanceExample[windowSize];
		InstanceExample[] windowForRanker=new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			InstanceExample InstForRanker = fsWithoutSA.nextInstance();
			windowForRanker[numberSamples % windowSize]=InstForRanker;
			numberSamples++;
			Instance tmpInst = trainInst.copy();
			tmpInst.setMissing(saIndex);
			double[] votes = learner.getVotesForInstance(tmpInst);
			evaluator.addResult(trainInstanceExample, votes);
			//training from beginning until bias occurs
			if (stopTrain==false)
				learner.trainOnInstance(trainInst);

			ranker.trainOnInstance(InstForRanker);

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
					changes=0;
					if (Math.abs(discClassifier)>epsilon){numberOfchanges += 1;
						//massaging
						int saNum = saPos+saNeg;
						int nSaNum= nSaPos+nSaNeg;
						if (discClassifier<-epsilon){//reverse discrimination & previous did the change							
							changes=((double)saPos*(double)nSaNum-(double)nSaPos*(double)saNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
							if (changes>0){
								if (saVal.equals("Female"))
									saVal="Male";
								else
									saVal="Female";
							}
						}else{
							changes=((double)nSaPos*(double)saNum-(double)saPos*(double)nSaNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
						}

						if (changes>0){ //massaging taking place
							learner.resetLearning();
							stopTrain=true;
							//predict on the current window
							for (int i=0; i<windowSize; i++){
								double[] ranker_votes = ranker.getVotesForInstance(windowForRanker[i]);
								ranker_evaluator.addResult(windowForRanker[i], ranker_votes);
							}
							int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
							int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();
//							String[] saValFromSortedScores=ranker_evaluator.getAucEstimator().getSAValFromSortedScores();
							double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
							String[] saValFromSortedScores=new String[windowSize];
							for (int i=0; i<windowSize; i++){
								String[] splits=windowList[posWindow[i]%windowSize].instance.toString().split(",");
								saValFromSortedScores[i]=splits[saIndex];
							}
							rankingWithSA(posWindow,saValFromSortedScores,sortedLabels,sortedScores);
							relabel_M(changes);
						}
						if (stopTrain==true){
							for (int i=0; i<windowSize; i++){
								learner.trainOnInstance(windowList[i]);
							}
						}
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
					writer.println(numberSamples+","+discData+","+discClassifier+","+changes+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);
				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numberOfchanges);
		writer.close();
		System.out.println("Done 7. ");
	}
	//model 8: RankerNoSA:Y	ResetRanker		ResetLearner	PredictNoSA:Y
	public static void NoResetLearnerNoResetRanker_PredictNoSA(String infileWithoutSA,String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		ArffFileStream fsWithoutSA = new ArffFileStream(infileWithoutSA, -1);
		fsWithoutSA.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		ranker.setModelContext(fsWithoutSA.getHeader());
		ranker.prepareForUse();
		WindowAUCImbalancedPerformanceEvaluator ranker_evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		ranker_evaluator.widthOption.setValue(windowSize);
		ranker_evaluator.setIndex(saIndex);
		ranker_evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,changes,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		double changes=0; int numberOfchanges = 0;
		windowList = new InstanceExample[windowSize];
		InstanceExample[] windowForRanker=new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			InstanceExample InstForRanker = fsWithoutSA.nextInstance();
			windowForRanker[numberSamples % windowSize]=InstForRanker;
			numberSamples++;
			Instance tmpInst = trainInst.copy();
			tmpInst.setMissing(saIndex);
			double[] votes = learner.getVotesForInstance(tmpInst);
			evaluator.addResult(trainInstanceExample, votes);
			//training from beginning until bias occurs
			if (stopTrain==false)
				learner.trainOnInstance(trainInst);

			ranker.trainOnInstance(InstForRanker);

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
					changes=0;
					if (Math.abs(discClassifier)>epsilon){numberOfchanges += 1;
						//massaging
						int saNum = saPos+saNeg;
						int nSaNum= nSaPos+nSaNeg;
						if (discClassifier<-epsilon){//reverse discrimination & previous did the change							
							changes=((double)saPos*(double)nSaNum-(double)nSaPos*(double)saNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
							if (changes>0){
								if (saVal.equals("Female"))
									saVal="Male";
								else
									saVal="Female";
							}
						}else{
							changes=((double)nSaPos*(double)saNum-(double)saPos*(double)nSaNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
						}

						if (changes>0){ //massaging taking place
							if (stopTrain==false){
								learner.resetLearning();
								stopTrain=true;
							}
							//predict on the current window
							for (int i=0; i<windowSize; i++){
								double[] ranker_votes = ranker.getVotesForInstance(windowForRanker[i]);
								ranker_evaluator.addResult(windowForRanker[i], ranker_votes);
							}
							int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
							int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();
//							String[] saValFromSortedScores=ranker_evaluator.getAucEstimator().getSAValFromSortedScores();
							double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
							String[] saValFromSortedScores=new String[windowSize];
							for (int i=0; i<windowSize; i++){
								String[] splits=windowList[posWindow[i]%windowSize].instance.toString().split(",");
								saValFromSortedScores[i]=splits[saIndex];
							}
							rankingWithSA(posWindow,saValFromSortedScores,sortedLabels,sortedScores);
							relabel_M(changes);
						}
						if (stopTrain==true){
							for (int i=0; i<windowSize; i++){
								learner.trainOnInstance(windowList[i]);
							}
						}
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
					writer.println(numberSamples+","+discData+","+discClassifier+","+changes+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);
				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numberOfchanges);
		writer.close();
		System.out.println("Done 8. ");
	}
	//model 9:  model 1+  LearnerCont.Learn: N
	public static void NoResetLearnerResetRanker_LearnerNoContLearn(String infileWithoutSA,String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		ArffFileStream fsWithoutSA = new ArffFileStream(infileWithoutSA, -1);
		fsWithoutSA.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,changes,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		double changes=0; int numberOfchanges = 0;
		windowList = new InstanceExample[windowSize];
		InstanceExample[] windowForRanker=new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			windowForRanker[numberSamples%windowSize]=fsWithoutSA.nextInstance();

			numberSamples++;
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);
			//training from beginning until bias occurs
			if (stopTrain==false)
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
					changes=0;
					if (Math.abs(discClassifier)>epsilon){numberOfchanges += 1;
						//massaging
						int saNum = saPos+saNeg;
						int nSaNum= nSaPos+nSaNeg;
						if (discClassifier<-epsilon){//reverse discrimination & previous did the change							
							changes=((double)saPos*(double)nSaNum-(double)nSaPos*(double)saNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
							if (changes>0){
								if (saVal.equals("Female"))
									saVal="Male";
								else
									saVal="Female";
							}
						}else{
							changes=((double)nSaPos*(double)saNum-(double)saPos*(double)nSaNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
						}

						if (changes>0){ //massaging taking place
							if (stopTrain==false){
								learner.resetLearning();
								stopTrain=true;
							}
							//ranker
							ranker = new NaiveBayes();
							ranker.setModelContext(fsWithoutSA.getHeader());
							ranker.prepareForUse();
							WindowAUCImbalancedPerformanceEvaluator ranker_evaluator =new WindowAUCImbalancedPerformanceEvaluator();
							ranker_evaluator.widthOption.setValue(windowSize);
							ranker_evaluator.setIndex(saIndex);
							ranker_evaluator.prepareForUse();

							for (int i=0; i<windowSize; i++){
								ranker.trainOnInstance(windowForRanker[i]);
							}
							for (int i=0; i<windowSize; i++){
								double[] ranker_votes = ranker.getVotesForInstance(windowForRanker[i]);
								ranker_evaluator.addResult(windowForRanker[i], ranker_votes);
							}
							int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
							int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();
//							String[] saValFromSortedScores=ranker_evaluator.getAucEstimator().getSAValFromSortedScores();
							double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
							String[] saValFromSortedScores=new String[windowSize];
							for (int i=0; i<windowSize; i++){
								String[] splits=windowList[posWindow[i]%windowSize].instance.toString().split(",");
								saValFromSortedScores[i]=splits[saIndex];
							}
							rankingWithSA(posWindow,saValFromSortedScores,sortedLabels,sortedScores);
							relabel_M(changes);
							for (int i=0; i<windowSize; i++){
								learner.trainOnInstance(windowList[i]);
							}
						}
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
					writer.println(numberSamples+","+discData+","+discClassifier+","+changes+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);
				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numberOfchanges);
		writer.close();
		System.out.println("Done 9.");
	}
	//model 10: model 2+  LearnerCont.Learn: N
	public static void ResetLearnerResetRanker_LearnerNoContLearn(String infileWithoutSA,String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		ArffFileStream fsWithoutSA = new ArffFileStream(infileWithoutSA, -1);
		fsWithoutSA.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,changes,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		double changes=0; int numberOfchanges = 0;
		windowList = new InstanceExample[windowSize];
		InstanceExample[] windowForRanker=new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			windowForRanker[numberSamples%windowSize]=fsWithoutSA.nextInstance();

			numberSamples++;
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);
			//training from beginning until bias occurs
			if (stopTrain==false)
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
					changes=0;
					if (Math.abs(discClassifier)>epsilon){numberOfchanges += 1;
						//massaging
						int saNum = saPos+saNeg;
						int nSaNum= nSaPos+nSaNeg;
						if (discClassifier<-epsilon){//reverse discrimination & previous did the change							
							changes=((double)saPos*(double)nSaNum-(double)nSaPos*(double)saNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
							if (changes>0){
								if (saVal.equals("Female"))
									saVal="Male";
								else
									saVal="Female";
							}
						}else{
							changes=((double)nSaPos*(double)saNum-(double)saPos*(double)nSaNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
						}

						if (changes>0){ //massaging taking place						
							stopTrain=true;
							//ranker
							ranker = new NaiveBayes();
							ranker.setModelContext(fsWithoutSA.getHeader());
							ranker.prepareForUse();
							WindowAUCImbalancedPerformanceEvaluator ranker_evaluator =new WindowAUCImbalancedPerformanceEvaluator();
							ranker_evaluator.widthOption.setValue(windowSize);
							ranker_evaluator.setIndex(saIndex);
							ranker_evaluator.prepareForUse();

							for (int i=0; i<windowSize; i++){
								ranker.trainOnInstance(windowForRanker[i]);
							}
							for (int i=0; i<windowSize; i++){
								double[] ranker_votes = ranker.getVotesForInstance(windowForRanker[i]);
								ranker_evaluator.addResult(windowForRanker[i], ranker_votes);
							}
							int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
							int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();
//							String[] saValFromSortedScores=ranker_evaluator.getAucEstimator().getSAValFromSortedScores();
							double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
							String[] saValFromSortedScores=new String[windowSize];
							for (int i=0; i<windowSize; i++){
								String[] splits=windowList[posWindow[i]%windowSize].instance.toString().split(",");
								saValFromSortedScores[i]=splits[saIndex];
							}
							rankingWithSA(posWindow,saValFromSortedScores,sortedLabels,sortedScores);
							relabel_M(changes);
							learner.resetLearning();
							for (int i=0; i<windowSize; i++){
								learner.trainOnInstance(windowList[i]);
							}
						}

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
					writer.println(numberSamples+","+discData+","+discClassifier+","+changes+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);
				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numberOfchanges);
		writer.close();
		System.out.println("Done 10. ");
	}
	//model 11: model 3+  LearnerCont.Learn: N
	public static void ResetLearnerNoResetRanker_LearnerNoContLearn(String infileWithoutSA,String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		ArffFileStream fsWithoutSA = new ArffFileStream(infileWithoutSA, -1);
		fsWithoutSA.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		ranker.setModelContext(fsWithoutSA.getHeader());
		ranker.prepareForUse();
		WindowAUCImbalancedPerformanceEvaluator ranker_evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		ranker_evaluator.widthOption.setValue(windowSize);
		ranker_evaluator.setIndex(saIndex);
		ranker_evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,changes,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		double changes=0; int numberOfchanges = 0;
		windowList = new InstanceExample[windowSize];
		InstanceExample[] windowForRanker=new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			InstanceExample InstForRanker = fsWithoutSA.nextInstance();
			windowForRanker[numberSamples % windowSize]=InstForRanker;
			numberSamples++;
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);
			//training from beginning until bias occurs
			if (stopTrain==false)
				learner.trainOnInstance(trainInst);

			ranker.trainOnInstance(InstForRanker);

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
					changes=0;
					if (Math.abs(discClassifier)>epsilon){numberOfchanges += 1;
						//massaging
						int saNum = saPos+saNeg;
						int nSaNum= nSaPos+nSaNeg;
						if (discClassifier<-epsilon){//reverse discrimination & previous did the change							
							changes=((double)saPos*(double)nSaNum-(double)nSaPos*(double)saNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
							if (changes>0){
								if (saVal.equals("Female"))
									saVal="Male";
								else
									saVal="Female";
							}
						}else{
							changes=((double)nSaPos*(double)saNum-(double)saPos*(double)nSaNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
						}

						if (changes>0){ //massaging taking place
							stopTrain=true;
							//predict on the current window
							for (int i=0; i<windowSize; i++){
								double[] ranker_votes = ranker.getVotesForInstance(windowForRanker[i]);
								ranker_evaluator.addResult(windowForRanker[i], ranker_votes);
							}
							int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
							int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();
//							String[] saValFromSortedScores=ranker_evaluator.getAucEstimator().getSAValFromSortedScores();
							double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
							String[] saValFromSortedScores=new String[windowSize];
							for (int i=0; i<windowSize; i++){
								String[] splits=windowList[posWindow[i]%windowSize].instance.toString().split(",");
								saValFromSortedScores[i]=splits[saIndex];
							}
							rankingWithSA(posWindow,saValFromSortedScores,sortedLabels,sortedScores);
							relabel_M(changes);
							learner.resetLearning();
							for (int i=0; i<windowSize; i++){
								learner.trainOnInstance(windowList[i]);
							}
						}
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
					writer.println(numberSamples+","+discData+","+discClassifier+","+changes+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);
				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numberOfchanges);
		writer.close();
		System.out.println("Done 11. ");
	}
	//model 12: model 4+  LearnerCont.Learn: N
	public static void NoResetLearnerNoResetRanker_LearnerNoContLearn(String infileWithoutSA,String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		ArffFileStream fsWithoutSA = new ArffFileStream(infileWithoutSA, -1);
		fsWithoutSA.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		ranker.setModelContext(fsWithoutSA.getHeader());
		ranker.prepareForUse();
		WindowAUCImbalancedPerformanceEvaluator ranker_evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		ranker_evaluator.widthOption.setValue(windowSize);
		ranker_evaluator.setIndex(saIndex);
		ranker_evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,changes,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		double changes=0; int numberOfchanges = 0;
		windowList = new InstanceExample[windowSize];
		InstanceExample[] windowForRanker=new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			InstanceExample InstForRanker = fsWithoutSA.nextInstance();
			windowForRanker[numberSamples % windowSize]=InstForRanker;
			numberSamples++;
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);
			//training from beginning until bias occurs
			if (stopTrain==false)
				learner.trainOnInstance(trainInst);

			ranker.trainOnInstance(InstForRanker);

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
					changes=0;
					if (Math.abs(discClassifier)>epsilon){numberOfchanges += 1;
						//massaging
						int saNum = saPos+saNeg;
						int nSaNum= nSaPos+nSaNeg;
						if (discClassifier<-epsilon){//reverse discrimination & previous did the change							
							changes=((double)saPos*(double)nSaNum-(double)nSaPos*(double)saNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
							if (changes>0){
								if (saVal.equals("Female"))
									saVal="Male";
								else
									saVal="Female";
							}
						}else{
							changes=((double)nSaPos*(double)saNum-(double)saPos*(double)nSaNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
						}

						if (changes>0){ //massaging taking place
							if (stopTrain==false){
								learner.resetLearning();
								stopTrain=true;
							}
							//predict on the current window
							for (int i=0; i<windowSize; i++){
								double[] ranker_votes = ranker.getVotesForInstance(windowForRanker[i]);
								ranker_evaluator.addResult(windowForRanker[i], ranker_votes);
							}
							int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
							int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();
//							String[] saValFromSortedScores=ranker_evaluator.getAucEstimator().getSAValFromSortedScores();
							double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
							String[] saValFromSortedScores=new String[windowSize];
							for (int i=0; i<windowSize; i++){
								String[] splits=windowList[posWindow[i]%windowSize].instance.toString().split(",");
								saValFromSortedScores[i]=splits[saIndex];
							}
							rankingWithSA(posWindow,saValFromSortedScores,sortedLabels,sortedScores);
							relabel_M(changes);
							for (int i=0; i<windowSize; i++){
								learner.trainOnInstance(windowList[i]);
							}
						}
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
					writer.println(numberSamples+","+discData+","+discClassifier+","+changes+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);
				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numberOfchanges);
		writer.close();
		System.out.println("Done 12. ");
	}
	//model 13: model 5+  LearnerCont.Learn: N
	public static void NoResetLearnerResetRanker_PredictNoSA_LearnerNoContLearn(String infileWithoutSA,String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		ArffFileStream fsWithoutSA = new ArffFileStream(infileWithoutSA, -1);
		fsWithoutSA.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,changes,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		double changes=0; int numberOfchanges = 0;
		windowList = new InstanceExample[windowSize];
		InstanceExample[] windowForRanker=new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			windowForRanker[numberSamples%windowSize]=fsWithoutSA.nextInstance();

			numberSamples++;
			Instance tmpInst = trainInst.copy();
			tmpInst.setMissing(saIndex);
			double[] votes = learner.getVotesForInstance(tmpInst);
			evaluator.addResult(trainInstanceExample, votes);
			//training from beginning until bias occurs
			if (stopTrain==false)
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
					changes=0;
					if (Math.abs(discClassifier)>epsilon){numberOfchanges += 1;
						//massaging
						int saNum = saPos+saNeg;
						int nSaNum= nSaPos+nSaNeg;
						if (discClassifier<-epsilon){//reverse discrimination & previous did the change							
							changes=((double)saPos*(double)nSaNum-(double)nSaPos*(double)saNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
							if (changes>0){
								if (saVal.equals("Female"))
									saVal="Male";
								else
									saVal="Female";
							}
						}else{
							changes=((double)nSaPos*(double)saNum-(double)saPos*(double)nSaNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
						}

						if (changes>0){ //massaging taking place
							if (stopTrain==false){
								learner.resetLearning();
								stopTrain=true;
							}
							//ranker
							ranker = new NaiveBayes();
							ranker.setModelContext(fsWithoutSA.getHeader());
							ranker.prepareForUse();
							WindowAUCImbalancedPerformanceEvaluator ranker_evaluator =new WindowAUCImbalancedPerformanceEvaluator();
							ranker_evaluator.widthOption.setValue(windowSize);
							ranker_evaluator.setIndex(saIndex);
							ranker_evaluator.prepareForUse();

							for (int i=0; i<windowSize; i++){
								ranker.trainOnInstance(windowForRanker[i]);
							}
							for (int i=0; i<windowSize; i++){
								double[] ranker_votes = ranker.getVotesForInstance(windowForRanker[i]);
								ranker_evaluator.addResult(windowForRanker[i], ranker_votes);
							}
							int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
							int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();
//							String[] saValFromSortedScores=ranker_evaluator.getAucEstimator().getSAValFromSortedScores();
							double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
							String[] saValFromSortedScores=new String[windowSize];
							for (int i=0; i<windowSize; i++){
								String[] splits=windowList[posWindow[i]%windowSize].instance.toString().split(",");
								saValFromSortedScores[i]=splits[saIndex];
							}
							rankingWithSA(posWindow,saValFromSortedScores,sortedLabels,sortedScores);
							relabel_M(changes);
							for (int i=0; i<windowSize; i++){
								learner.trainOnInstance(windowList[i]);
							}
						}
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
					writer.println(numberSamples+","+discData+","+discClassifier+","+changes+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);
				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numberOfchanges);
		writer.close();
		System.out.println("Done 13.");
	}
	//model 14: model 6+  LearnerCont.Learn: N
	public static void ResetLearnerResetRanker_PredictNoSA_LearnerNoContLearn(String infileWithoutSA,String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		ArffFileStream fsWithoutSA = new ArffFileStream(infileWithoutSA, -1);
		fsWithoutSA.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,changes,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		double changes=0; int numberOfchanges = 0;
		windowList = new InstanceExample[windowSize];
		InstanceExample[] windowForRanker=new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			windowForRanker[numberSamples%windowSize]=fsWithoutSA.nextInstance();

			numberSamples++;
			Instance tmpInst = trainInst.copy();
			tmpInst.setMissing(saIndex);
			double[] votes = learner.getVotesForInstance(tmpInst);
			evaluator.addResult(trainInstanceExample, votes);
			//training from beginning until bias occurs
			if (stopTrain==false)
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
					changes=0;
					if (Math.abs(discClassifier)>epsilon){numberOfchanges += 1;
						//massaging
						int saNum = saPos+saNeg;
						int nSaNum= nSaPos+nSaNeg;
						if (discClassifier<-epsilon){//reverse discrimination & previous did the change							
							changes=((double)saPos*(double)nSaNum-(double)nSaPos*(double)saNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
							if (changes>0){
								if (saVal.equals("Female"))
									saVal="Male";
								else
									saVal="Female";
							}
						}else{
							changes=((double)nSaPos*(double)saNum-(double)saPos*(double)nSaNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
						}

						if (changes>0){ //massaging taking place
							stopTrain=true;

							//ranker
							ranker = new NaiveBayes();
							ranker.setModelContext(fsWithoutSA.getHeader());
							ranker.prepareForUse();
							WindowAUCImbalancedPerformanceEvaluator ranker_evaluator =new WindowAUCImbalancedPerformanceEvaluator();
							ranker_evaluator.widthOption.setValue(windowSize);
							ranker_evaluator.setIndex(saIndex);
							ranker_evaluator.prepareForUse();

							for (int i=0; i<windowSize; i++){
								ranker.trainOnInstance(windowForRanker[i]);
							}
							for (int i=0; i<windowSize; i++){
								double[] ranker_votes = ranker.getVotesForInstance(windowForRanker[i]);
								ranker_evaluator.addResult(windowForRanker[i], ranker_votes);
							}
							int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
							int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();
//							String[] saValFromSortedScores=ranker_evaluator.getAucEstimator().getSAValFromSortedScores();
							double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
							String[] saValFromSortedScores=new String[windowSize];
							for (int i=0; i<windowSize; i++){
								String[] splits=windowList[posWindow[i]%windowSize].instance.toString().split(",");
								saValFromSortedScores[i]=splits[saIndex];
							}
							rankingWithSA(posWindow,saValFromSortedScores,sortedLabels,sortedScores);
							relabel_M(changes);
							learner.resetLearning();
							for (int i=0; i<windowSize; i++){
								learner.trainOnInstance(windowList[i]);
							}
						}

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
					writer.println(numberSamples+","+discData+","+discClassifier+","+changes+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);
				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numberOfchanges);
		writer.close();
		System.out.println("Done 14. ");
	}
	//model 15: model 7+  LearnerCont.Learn: N
	public static void ResetLearnerNoResetRanker_PredictNoSA_LearnerNoContLearn(String infileWithoutSA,String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		ArffFileStream fsWithoutSA = new ArffFileStream(infileWithoutSA, -1);
		fsWithoutSA.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		ranker.setModelContext(fsWithoutSA.getHeader());
		ranker.prepareForUse();
		WindowAUCImbalancedPerformanceEvaluator ranker_evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		ranker_evaluator.widthOption.setValue(windowSize);
		ranker_evaluator.setIndex(saIndex);
		ranker_evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,changes,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		double changes=0; int numberOfchanges = 0;
		windowList = new InstanceExample[windowSize];
		InstanceExample[] windowForRanker=new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			InstanceExample InstForRanker = fsWithoutSA.nextInstance();
			windowForRanker[numberSamples % windowSize]=InstForRanker;
			numberSamples++;
			Instance tmpInst = trainInst.copy();
			tmpInst.setMissing(saIndex);
			double[] votes = learner.getVotesForInstance(tmpInst);
			evaluator.addResult(trainInstanceExample, votes);
			//training from beginning until bias occurs
			if (stopTrain==false)
				learner.trainOnInstance(trainInst);

			ranker.trainOnInstance(InstForRanker);

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
					changes=0;
					if (Math.abs(discClassifier)>epsilon){numberOfchanges += 1;
						//massaging
						int saNum = saPos+saNeg;
						int nSaNum= nSaPos+nSaNeg;
						if (discClassifier<-epsilon){//reverse discrimination & previous did the change							
							changes=((double)saPos*(double)nSaNum-(double)nSaPos*(double)saNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
							if (changes>0){
								if (saVal.equals("Female"))
									saVal="Male";
								else
									saVal="Female";
							}
						}else{
							changes=((double)nSaPos*(double)saNum-(double)saPos*(double)nSaNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
						}

						if (changes>0){ //massaging taking place
							learner.resetLearning();
							stopTrain=true;
							//predict on the current window
							for (int i=0; i<windowSize; i++){
								double[] ranker_votes = ranker.getVotesForInstance(windowForRanker[i]);
								ranker_evaluator.addResult(windowForRanker[i], ranker_votes);
							}
							int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
							int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();
//							String[] saValFromSortedScores=ranker_evaluator.getAucEstimator().getSAValFromSortedScores();
							double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
							String[] saValFromSortedScores=new String[windowSize];
							for (int i=0; i<windowSize; i++){
								String[] splits=windowList[posWindow[i]%windowSize].instance.toString().split(",");
								saValFromSortedScores[i]=splits[saIndex];
							}
							rankingWithSA(posWindow,saValFromSortedScores,sortedLabels,sortedScores);
							relabel_M(changes);
							for (int i=0; i<windowSize; i++){
								learner.trainOnInstance(windowList[i]);
							}

						}
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
					writer.println(numberSamples+","+discData+","+discClassifier+","+changes+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);
				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numberOfchanges);
		writer.close();
		System.out.println("Done 15. ");
	}
	//model 16: model 8+  LearnerCont.Learn: N
	public static void NoResetLearnerNoResetRanker_PredictNoSA_LearnerNoContLearn(String infileWithoutSA,String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		ArffFileStream fsWithoutSA = new ArffFileStream(infileWithoutSA, -1);
		fsWithoutSA.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		ranker.setModelContext(fsWithoutSA.getHeader());
		ranker.prepareForUse();
		WindowAUCImbalancedPerformanceEvaluator ranker_evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		ranker_evaluator.widthOption.setValue(windowSize);
		ranker_evaluator.setIndex(saIndex);
		ranker_evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,changes,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		double changes=0; int numberOfchanges = 0;
		windowList = new InstanceExample[windowSize];
		InstanceExample[] windowForRanker=new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			InstanceExample InstForRanker = fsWithoutSA.nextInstance();
			windowForRanker[numberSamples % windowSize]=InstForRanker;
			numberSamples++;
			Instance tmpInst = trainInst.copy();
			tmpInst.setMissing(saIndex);
			double[] votes = learner.getVotesForInstance(tmpInst);
			evaluator.addResult(trainInstanceExample, votes);
			//training from beginning until bias occurs
			if (stopTrain==false)
				learner.trainOnInstance(trainInst);

			ranker.trainOnInstance(InstForRanker);

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
					changes=0;
					if (Math.abs(discClassifier)>epsilon){numberOfchanges += 1;
						//massaging
						int saNum = saPos+saNeg;
						int nSaNum= nSaPos+nSaNeg;
						if (discClassifier<-epsilon){//reverse discrimination & previous did the change							
							changes=((double)saPos*(double)nSaNum-(double)nSaPos*(double)saNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
							if (changes>0){
								if (saVal.equals("Female"))
									saVal="Male";
								else
									saVal="Female";
							}
						}else{
							changes=((double)nSaPos*(double)saNum-(double)saPos*(double)nSaNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
						}

						if (changes>0){ //massaging taking place
							if (stopTrain==false){
								learner.resetLearning();
								stopTrain=true;
							}
							//predict on the current window
							for (int i=0; i<windowSize; i++){
								double[] ranker_votes = ranker.getVotesForInstance(windowForRanker[i]);
								ranker_evaluator.addResult(windowForRanker[i], ranker_votes);
							}
							int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
							int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();
//							String[] saValFromSortedScores=ranker_evaluator.getAucEstimator().getSAValFromSortedScores();
							double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
							String[] saValFromSortedScores=new String[windowSize];
							for (int i=0; i<windowSize; i++){
								String[] splits=windowList[posWindow[i]%windowSize].instance.toString().split(",");
								saValFromSortedScores[i]=splits[saIndex];
							}
							rankingWithSA(posWindow,saValFromSortedScores,sortedLabels,sortedScores);
							relabel_M(changes);
							for (int i=0; i<windowSize; i++){
								learner.trainOnInstance(windowList[i]);
							}
						}
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
					writer.println(numberSamples+","+discData+","+discClassifier+","+changes+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);
				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numberOfchanges);
		writer.close();
		System.out.println("Done 16. ");
	}
	public static void KeepReseting(String filename, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(filename, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();
		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();
		ranker.setModelContext(fs.getHeader());
		ranker.prepareForUse();
		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		int numberSamples=0;int cnt=0;
		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,changes, Acc, P, R, F1, rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		windowList = new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		int numberOfchanges = 0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			numberSamples++;
			//evaluator window
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);
			learner.trainOnInstance(trainInst);
			if (numberSamples>=windowSize){
				if (numberSamples%windowSize==0){
					cnt++;
					String[] labels;
					double[] predictions;
					int[] trueLabels;
					labels = evaluator.getAucEstimator().getSAVal();
					predictions = evaluator.getAucEstimator().getPredictions();
					trueLabels = evaluator.getAucEstimator().getTrueLabel();
					double discClassifier=DiscriminationScore(labels,predictions,trueLabels);
					avgDisc+=Math.abs(discClassifier);
					double discData = Disc_Data();
					if (Math.abs(discClassifier)>epsilon){
						numberOfchanges += 1;
						learner.resetLearning();
						for (int i=0; i<windowSize; i++){
							learner.trainOnInstance(windowList[i]);
						}
					}

					//evaluation
					double precision= evaluator.getAucEstimator().getPrecision();
					double acc = evaluator.getAucEstimator().getAccuracy();
					avgAcc+=acc;
					double recall=evaluator.getAucEstimator().getRecall();
					double f1=0;
					if (precision!=0||recall!=0)
						f1=2*precision*recall/(precision+recall);
					avgF1+=f1;
					double rocArea= evaluator.getAucEstimator().getAUC();
					avgROC+=rocArea;
					double prArea=evaluator.getAucEstimator().prArea();
					avgPR+=prArea;
					double kappa=evaluator.getAucEstimator().getKappa();
					avgKappa+=kappa;
					double kappaM=evaluator.getAucEstimator().getKappaM();
					avgKappaM+=kappaM;
					double gMean=evaluator.getAucEstimator().getGMean();
					avgGMean+=gMean;
					writer.println(numberSamples+","+discData+","+discClassifier+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);
				}
			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numberOfchanges);
		writer.close();
		System.out.println("DONE BASELINE KEEP RESETING WHEN GETTING BIAS");
	}
	//baseline: only do Msg in the 1st window bias
	public static void Massaging1stWindow(String filename, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(filename, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();
		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();
		ranker.setModelContext(fs.getHeader());
		ranker.prepareForUse();
		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();
		//evaluator for ranker
		WindowAUCImbalancedPerformanceEvaluator ranker_evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		ranker_evaluator.widthOption.setValue(windowSize);
		ranker_evaluator.setIndex(saIndex);
		ranker_evaluator.prepareForUse();
		int numberSamples=0;int cnt=0;
		double changes=0; int numberOfchanges = 0;
		boolean done=false;
		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,changes, Acc, P, R, F1, rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		windowList = new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			numberSamples++;
			//evaluator window
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);
			learner.trainOnInstance(trainInst);
			if (numberSamples>=windowSize){
				if (numberSamples%windowSize==0){
					cnt++;
					String[] labels;
					double[] predictions;
					int[] trueLabels;
					labels = evaluator.getAucEstimator().getSAVal();
					predictions = evaluator.getAucEstimator().getPredictions();
					trueLabels = evaluator.getAucEstimator().getTrueLabel();
					double discClassifier=DiscriminationScore(labels,predictions,trueLabels);
					avgDisc+=Math.abs(discClassifier);
					double discData = Disc_Data();
					if (Math.abs(discClassifier)>epsilon && done==false){
						//massaging
						int saNum = saPos+saNeg;
						int nSaNum= nSaPos+nSaNeg;
						changes=((double)nSaPos*(double)saNum-(double)saPos*(double)nSaNum
								-(double)(0/100)*(double)saNum*(double)nSaNum)
								/(double)(windowSize);
						if (changes>0){
							for (int i=0; i<windowSize; i++){
								ranker.trainOnInstance(windowList[i]);
							}
							for (int i=0; i<windowSize; i++){
								double[] ranker_votes = ranker.getVotesForInstance(windowList[i]);
								ranker_evaluator.addResult(windowList[i], ranker_votes);
							}
							//massaging
							int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
							int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();
							String[] saValFromSortedScores=ranker_evaluator.getAucEstimator().getSAValFromSortedScores();
							double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
							rankingWithSA(posWindow,saValFromSortedScores,sortedLabels,sortedScores);
							relabel_M(changes);

							learner.resetLearning();
							for (int i=0; i<windowSize; i++){
								learner.trainOnInstance(windowList[i]);
							}
							done=true;
						}
					}//end of comparing epsilon

					//evaluation
					double precision= evaluator.getAucEstimator().getPrecision();
					double acc = evaluator.getAucEstimator().getAccuracy();
					avgAcc+=acc;
					double recall=evaluator.getAucEstimator().getRecall();
					double f1=0;
					if (precision!=0||recall!=0)
						f1=2*precision*recall/(precision+recall);
					avgF1+=f1;
					double rocArea= evaluator.getAucEstimator().getAUC();
					avgROC+=rocArea;
					double prArea=evaluator.getAucEstimator().prArea();
					avgPR+=prArea;
					double kappa=evaluator.getAucEstimator().getKappa();
					avgKappa+=kappa;
					double kappaM=evaluator.getAucEstimator().getKappaM();
					avgKappaM+=kappaM;
					double gMean=evaluator.getAucEstimator().getGMean();
					avgGMean+=gMean;
					writer.println(numberSamples+","+discData+","+discClassifier+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);
				}
			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numberOfchanges);
		writer.close();
		System.out.println("DONE BASELINE MASSAGING 1ST WINDOW");
	}
	//baseline: remove SA from dataset
	public static void batchRemoveSA(String infileWithoutSA, String infileWithSA, String outfile) throws FileNotFoundException {
		ArffFileStream fs = new ArffFileStream(infileWithoutSA, -1);
		ArffFileStream fsWithSA = new ArffFileStream(infileWithSA, -1);
		for (int i=0; i<fsWithSA.getHeader().numAttributes(); i++){
			if (fsWithSA.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fsWithSA.prepareForUse();
		fs.prepareForUse();
		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,changes, Acc, P, R, F1, rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0;

		windowList = new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		int cnt=0;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;

			InstanceExample trainInstanceExample_WithSA = fsWithSA.nextInstance();
			windowList[numberSamples % windowSize]= trainInstanceExample_WithSA;
			numberSamples++;
			//evaluator window
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);
			learner.trainOnInstance(trainInst);
			if (numberSamples>=windowSize){
				if (numberSamples%windowSize==0){
					cnt++;
					//for calculation of discrimination score
					double[] predictions = evaluator.getAucEstimator().getPredictions();
					int[] trueLabels = evaluator.getAucEstimator().getTrueLabel();
					double discClassifier=DiscriminationScore_WithoutSA(predictions,trueLabels);
					avgDisc+=Math.abs(discClassifier);
					double discData = Disc_Data();

					double precision= evaluator.getAucEstimator().getPrecision();
					double acc = evaluator.getAucEstimator().getAccuracy();
					avgAcc+=acc;
					double recall=evaluator.getAucEstimator().getRecall();
					double f1=0;
					if (precision!=0||recall!=0)
						f1=2*precision*recall/(precision+recall);
					avgF1+=f1;
					double rocArea= evaluator.getAucEstimator().getAUC();
					avgROC+=rocArea;
					double prArea=evaluator.getAucEstimator().prArea();
					avgPR+=prArea;
					double kappa=evaluator.getAucEstimator().getKappa();
					avgKappa+=kappa;
					double kappaM=evaluator.getAucEstimator().getKappaM();
					avgKappaM+=kappaM;
					double gMean=evaluator.getAucEstimator().getGMean();
					avgGMean+=gMean;
					writer.println(numberSamples+","+discData+","+discClassifier+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);
				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt);
		writer.close();
		System.out.println("Done Baseline: Remove SA");
	}
	//model 1
	public static void RankerSALearnerSA(String infile, String outfile) throws FileNotFoundException{
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
		writer.println("noSamples, DiscData, DiscClassifier,changes,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		double changes=0; int numberOfchanges = 0;
		windowList = new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			numberSamples++;
			//evaluator window
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);
			//training from beginning until bias occurs
			if (stopTrain==false)
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
					changes=0;
					if (Math.abs(discClassifier)>epsilon){numberOfchanges += 1;
						//massaging
						int saNum = saPos+saNeg;
						int nSaNum= nSaPos+nSaNeg;
						if (discClassifier<-epsilon){//reverse discrimination & previous did the change							
							changes=((double)saPos*(double)nSaNum-(double)nSaPos*(double)saNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
							if (changes>0){
								if (saVal.equals("Female"))
									saVal="Male";
								else
									saVal="Female";
							}
						}else{
							changes=((double)nSaPos*(double)saNum-(double)saPos*(double)nSaNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
						}

						if (changes>0){ //massaging taking place
							if (stopTrain==false){
								learner.resetLearning();
								stopTrain=true;
							}
							//ranker
							ranker = new NaiveBayes();
							ranker.setModelContext(fs.getHeader());
							ranker.prepareForUse();
							WindowAUCImbalancedPerformanceEvaluator ranker_evaluator =new WindowAUCImbalancedPerformanceEvaluator();
							ranker_evaluator.widthOption.setValue(windowSize);
							ranker_evaluator.setIndex(saIndex);
							ranker_evaluator.prepareForUse();

							for (int i=0; i<windowSize; i++){
								ranker.trainOnInstance(windowList[i]);
							}
							for (int i=0; i<windowSize; i++){
								double[] ranker_votes = ranker.getVotesForInstance(windowList[i]);
								ranker_evaluator.addResult(windowList[i], ranker_votes);
							}
							int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
							int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();
							String[] saValFromSortedScores=ranker_evaluator.getAucEstimator().getSAValFromSortedScores();
							double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
							rankingWithSA(posWindow,saValFromSortedScores,sortedLabels,sortedScores);
							relabel_M(changes);
						}
						if (stopTrain==true){
							for (int i=0; i<windowSize; i++){
								learner.trainOnInstance(windowList[i]);
							}
						}
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
					writer.println(numberSamples+","+discData+","+discClassifier+","+changes+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);
				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numberOfchanges);
		writer.close();
		System.out.println("Done 1.New Ranker with SA");
	}
	//model 2
	public static void OldRankerSALearnerSA(String infile, String outfile) throws FileNotFoundException{
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
		ranker.setModelContext(fs.getHeader());
		ranker.prepareForUse();
		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();
		//evaluator for ranker
		WindowAUCImbalancedPerformanceEvaluator ranker_evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		ranker_evaluator.widthOption.setValue(windowSize);
		ranker_evaluator.setIndex(saIndex);
		ranker_evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,changes,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		double changes=0; int numberOfchanges = 0;
		windowList = new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			numberSamples++;
			//evaluator window
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);
			//training from beginning until bias occurs
			if (stopTrain==false)
				learner.trainOnInstance(trainInst);

			ranker.trainOnInstance(trainInst);
			double[] ranker_votes = ranker.getVotesForInstance(trainInstanceExample);
			ranker_evaluator.addResult(trainInstanceExample, ranker_votes);
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
					changes=0;
					if (Math.abs(discClassifier)>epsilon){numberOfchanges += 1;
						//massaging
						int saNum = saPos+saNeg;
						int nSaNum= nSaPos+nSaNeg;
						if (discClassifier<-epsilon){//reverse discrimination & previous did the change							
							changes=((double)saPos*(double)nSaNum-(double)nSaPos*(double)saNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
							if (changes>0){
								if (saVal.equals("Female"))
									saVal="Male";
								else
									saVal="Female";
							}
						}else{
							changes=((double)nSaPos*(double)saNum-(double)saPos*(double)nSaNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
						}

						if (changes>0){ //massaging taking place
							if (stopTrain==false){
								learner.resetLearning();
								stopTrain=true;
							}

							int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
							int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();
							String[] saValFromSortedScores=ranker_evaluator.getAucEstimator().getSAValFromSortedScores();
							double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
							rankingWithSA(posWindow,saValFromSortedScores,sortedLabels,sortedScores);
							relabel_M(changes);
						}
						if (stopTrain==true){
							for (int i=0; i<windowSize; i++){
								learner.trainOnInstance(windowList[i]);
							}
						}
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
					writer.println(numberSamples+","+discData+","+discClassifier+","+changes+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);
				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numberOfchanges);
		writer.close();
		System.out.println("Done 1.New Ranker with SA");
	}
	//model 3
	public static void RankerSALearnerNoSA(String infile, String outfile) throws FileNotFoundException{
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
		writer.println("noSamples, DiscData, DiscClassifier,changes,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		double changes=0; int numberOfchanges = 0;
		windowList = new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			numberSamples++;
			Instance tmpInst = trainInst.copy();
			tmpInst.setMissing(saIndex);
			double[] votes = learner.getVotesForInstance(tmpInst);
			evaluator.addResult(trainInstanceExample, votes);
			//training from beginning until bias occurs
			if (stopTrain==false){
				learner.trainOnInstance(trainInst);
			}
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
					changes=0;
					if (Math.abs(discClassifier)>epsilon){numberOfchanges += 1;
						//massaging
						int saNum = saPos+saNeg;
						int nSaNum= nSaPos+nSaNeg;
						if (discClassifier<-epsilon){//reverse discrimination & previous did the change							
							changes=((double)saPos*(double)nSaNum-(double)nSaPos*(double)saNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
							if (changes>0){
								if (saVal.equals("Female"))
									saVal="Male";
								else
									saVal="Female";
							}
						}else{
							changes=((double)nSaPos*(double)saNum-(double)saPos*(double)nSaNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
						}

						if (changes>0){ //massaging taking place
							if (stopTrain==false){
								learner.resetLearning();
								stopTrain=true;
							}
							//ranker
							ranker = new NaiveBayes();
							ranker.setModelContext(fs.getHeader());
							ranker.prepareForUse();
							WindowAUCImbalancedPerformanceEvaluator ranker_evaluator =new WindowAUCImbalancedPerformanceEvaluator();
							ranker_evaluator.widthOption.setValue(windowSize);
							ranker_evaluator.setIndex(saIndex);
							ranker_evaluator.prepareForUse();

							for (int i=0; i<windowSize; i++){
								ranker.trainOnInstance(windowList[i]);
							}
							for (int i=0; i<windowSize; i++){
								double[] ranker_votes = ranker.getVotesForInstance(windowList[i]);
								ranker_evaluator.addResult(windowList[i], ranker_votes);
							}
							int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
							int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();
							String[] saValFromSortedScores=ranker_evaluator.getAucEstimator().getSAValFromSortedScores();
							double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
							rankingWithSA(posWindow,saValFromSortedScores,sortedLabels,sortedScores);
							relabel_M(changes);
						}
						if (stopTrain==true){
							for (int i=0; i<windowSize; i++){
								learner.trainOnInstance(windowList[i]);
							}
						}
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
					writer.println(numberSamples+","+discData+","+discClassifier+","+changes+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);
				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numberOfchanges);
		writer.close();
		System.out.println("Done 3.New Ranker SA, learner withoutSA prediction");
	}
	//model 4
	public static void RankerWithoutSALearnerSA(String infileWithoutSA,String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		ArffFileStream fsWithoutSA = new ArffFileStream(infileWithoutSA, -1);
		fsWithoutSA.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,changes,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		double changes=0; int numberOfchanges = 0;
		windowList = new InstanceExample[windowSize];
		InstanceExample[] windowForRanker=new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			windowForRanker[numberSamples%windowSize]=fsWithoutSA.nextInstance();

			numberSamples++;
			double[] votes = learner.getVotesForInstance(trainInst);
			evaluator.addResult(trainInstanceExample, votes);
			//training from beginning until bias occurs
			if (stopTrain==false)
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
					changes=0;
					if (Math.abs(discClassifier)>epsilon){numberOfchanges += 1;
						//massaging
						int saNum = saPos+saNeg;
						int nSaNum= nSaPos+nSaNeg;
						if (discClassifier<-epsilon){//reverse discrimination & previous did the change							
							changes=((double)saPos*(double)nSaNum-(double)nSaPos*(double)saNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
							if (changes>0){
								if (saVal.equals("Female"))
									saVal="Male";
								else
									saVal="Female";
							}
						}else{
							changes=((double)nSaPos*(double)saNum-(double)saPos*(double)nSaNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
						}

						if (changes>0){ //massaging taking place
							if (stopTrain==false){
								learner.resetLearning();
								stopTrain=true;
							}
							//ranker
							ranker = new NaiveBayes();
							ranker.setModelContext(fsWithoutSA.getHeader());
							ranker.prepareForUse();
							WindowAUCImbalancedPerformanceEvaluator ranker_evaluator =new WindowAUCImbalancedPerformanceEvaluator();
							ranker_evaluator.widthOption.setValue(windowSize);
							ranker_evaluator.setIndex(saIndex);
							ranker_evaluator.prepareForUse();

							for (int i=0; i<windowSize; i++){
								ranker.trainOnInstance(windowForRanker[i]);
							}
							for (int i=0; i<windowSize; i++){
								double[] ranker_votes = ranker.getVotesForInstance(windowForRanker[i]);
								ranker_evaluator.addResult(windowForRanker[i], ranker_votes);
							}
							int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
							int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();
//							String[] saValFromSortedScores=ranker_evaluator.getAucEstimator().getSAValFromSortedScores();
							double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
							String[] saValFromSortedScores=new String[windowSize];
							for (int i=0; i<windowSize; i++){
								String[] splits=windowList[posWindow[i]%windowSize].instance.toString().split(",");
								saValFromSortedScores[i]=splits[saIndex];
							}
							rankingWithSA(posWindow,saValFromSortedScores,sortedLabels,sortedScores);
							relabel_M(changes);
						}
						if (stopTrain==true){
							for (int i=0; i<windowSize; i++){
								learner.trainOnInstance(windowList[i]);
							}
						}
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
					writer.println(numberSamples+","+discData+","+discClassifier+","+changes+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);
				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numberOfchanges);
		writer.close();
		System.out.println("Done 4.New Ranker withou SA, learner");
	}
	//model 5
	public static void RankerWithoutSALearnerWithoutSA(String infileWithoutSA,String infile, String outfile) throws FileNotFoundException{
		ArffFileStream fs = new ArffFileStream(infile, -1);
		for (int i=0; i<fs.getHeader().numAttributes(); i++){
			if (fs.getHeader().attribute(i).name().equals(saName)){
				saIndex = i;
				break;
			}
		}
		fs.prepareForUse();

		ArffFileStream fsWithoutSA = new ArffFileStream(infileWithoutSA, -1);
		fsWithoutSA.prepareForUse();

		learner.setModelContext(fs.getHeader());
		learner.prepareForUse();

		//evaluator definition
		WindowAUCImbalancedPerformanceEvaluator evaluator =new WindowAUCImbalancedPerformanceEvaluator();
		evaluator.widthOption.setValue(windowSize);
		evaluator.setIndex(saIndex);
		evaluator.prepareForUse();

		PrintWriter writer = new PrintWriter(outfile);
		writer.println("noSamples, DiscData, DiscClassifier,changes,Acc,P,R,F1,rocArea, prArea,kappa,kappaM,GMean");
		saPos=0; saNeg=0; nSaPos=0; nSaNeg=0;//true labels
		int numberSamples=0; int cnt=0;
		double changes=0; int numberOfchanges = 0;
		windowList = new InstanceExample[windowSize];
		InstanceExample[] windowForRanker=new InstanceExample[windowSize];
		double avgDisc=0, avgAcc=0, avgF1=0, avgROC=0, avgPR=0;
		double avgKappa=0, avgKappaM=0, avgGMean=0;
		boolean stopTrain=false;
		while (fs.hasMoreInstances()){
			InstanceExample trainInstanceExample = fs.nextInstance();
			Instance trainInst = trainInstanceExample.instance;
			windowList[numberSamples % windowSize]= trainInstanceExample;
			windowForRanker[numberSamples%windowSize]=fsWithoutSA.nextInstance();

			numberSamples++;
			Instance tmpInst = trainInst.copy();
			tmpInst.setMissing(saIndex);
			double[] votes = learner.getVotesForInstance(tmpInst);
			evaluator.addResult(trainInstanceExample, votes);
			//training from beginning until bias occurs
			if (stopTrain==false){
				learner.trainOnInstance(trainInst);
			}
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
					changes=0;
					if (Math.abs(discClassifier)>epsilon){numberOfchanges += 1;
						//massaging
						int saNum = saPos+saNeg;
						int nSaNum= nSaPos+nSaNeg;
						if (discClassifier<-epsilon){//reverse discrimination & previous did the change							
							changes=((double)saPos*(double)nSaNum-(double)nSaPos*(double)saNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
							if (changes>0){
								if (saVal.equals("Female"))
									saVal="Male";
								else
									saVal="Female";
							}
						}else{
							changes=((double)nSaPos*(double)saNum-(double)saPos*(double)nSaNum
									-(double)(0/100)*(double)saNum*(double)nSaNum)
									/(double)(windowSize);
						}

						if (changes>0){ //massaging taking place
							if (stopTrain==false){
								learner.resetLearning();
								stopTrain=true;
							}
							//ranker
							ranker = new NaiveBayes();
							ranker.setModelContext(fsWithoutSA.getHeader());
							ranker.prepareForUse();
							WindowAUCImbalancedPerformanceEvaluator ranker_evaluator =new WindowAUCImbalancedPerformanceEvaluator();
							ranker_evaluator.widthOption.setValue(windowSize);
							ranker_evaluator.setIndex(saIndex);
							ranker_evaluator.prepareForUse();

							for (int i=0; i<windowSize; i++){
								ranker.trainOnInstance(windowForRanker[i]);
							}
							for (int i=0; i<windowSize; i++){
								double[] ranker_votes = ranker.getVotesForInstance(windowForRanker[i]);
								ranker_evaluator.addResult(windowForRanker[i], ranker_votes);
							}
							int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
							int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();
//							String[] saValFromSortedScores=ranker_evaluator.getAucEstimator().getSAValFromSortedScores();
							double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
							String[] saValFromSortedScores=new String[windowSize];
							for (int i=0; i<windowSize; i++){
								String[] splits=windowList[posWindow[i]%windowSize].instance.toString().split(",");
								saValFromSortedScores[i]=splits[saIndex];
							}
							rankingWithSA(posWindow,saValFromSortedScores,sortedLabels,sortedScores);
							relabel_M(changes);
						}
						if (stopTrain==true){
							for (int i=0; i<windowSize; i++){
								learner.trainOnInstance(windowList[i]);
							}
						}
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
					writer.println(numberSamples+","+discData+","+discClassifier+","+changes+","+acc+","
							+precision+","+recall+","+f1+","+rocArea+","+prArea+","+kappa+","+kappaM+","+gMean);
				}

			}
		}
		writer.println("AvgDisc,AvgAcc,AvgF1,AvgROC,AvgPR,AvgKappa,AvgKappaM,AvgGMean,changes");
		writer.println(avgDisc/(double)cnt+","+avgAcc/(double)cnt+","+avgF1/(double)cnt+","+avgROC/(double)cnt+","+avgPR/(double)cnt+","+avgKappa/(double)cnt+","+avgKappaM/(double)(numberSamples-windowSize+1)+","+avgGMean/(double)cnt + "," + numberOfchanges);
		writer.close();
		System.out.println("Done 5.New Ranker withou SA, learner Missing Value");
	}

	public static void rankingWithSA(int[] posWindow, String[] saValFromSortedScores,int[] sortedLabels, double[] sortedScores){
		double[][] promotionList=new double[windowSize][2];
		double[][] demotionList=new double[windowSize][2];
		int demote = 0, promote = 0;
		for (int i=0; i<posWindow.length; i++){
			String sa = saValFromSortedScores[i];
			int classVal = sortedLabels[i];

            if (sa.equals(saVal) && classVal==notDesiredClass){
                promotionList[promote][0]=posWindow[i]%windowSize;
				promotionList[promote++][1]=sortedScores[i];
			}else if (!sa.equals(saVal) && classVal==desiredClass){
				demotionList[demote][0]=posWindow[i]%windowSize;
				demotionList[demote++][1]=sortedScores[i];
			}
		}//end of for i
//        System.out.println("promotionList.length = " + promotionList.length);

        sortedPromotionList = sorting(promotionList, promote, 1);
		sortedDemotionList = sorting(demotionList, demote, 2);
//		System.out.println("sortedPromotionList.length = " + sortedPromotionList.length);
//		System.out.println("sortedDemotionList.length = " + sortedDemotionList.length);
	}

	public static void relabel_M(double changes){

        if (changes > sortedDemotionList.length || changes > sortedPromotionList.length){
            changes = Math.min(sortedDemotionList.length ,sortedPromotionList.length);
        }

		for (int i=0; i<changes; i++){
			int index=0;
			index = (int)sortedPromotionList[i][0];
			windowList[index].instance.setClassValue(desiredClass);

			index = (int)sortedDemotionList[i][0];
			windowList[index].instance.setClassValue(notDesiredClass);
		}
	}
	public static double DiscriminationScore(String[] labels, double[] predictions, int[] trueLabels){
		tpDeprived=0;tnDeprived=0;fnDeprived=0;fpDeprived=0;
		tpFavored=0;tnFavored=0;fnFavored=0;fpFavored=0;
		for (int i=0; i<windowSize; i++){
			if (labels[i].equals(saVal)){ //Deprived
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
		if ((saPos+saNeg)==0)
			return 100*(double)(tpFavored+fpFavored)/(double)(nSaPos+nSaNeg);
		else{
			if ((nSaPos+nSaNeg)==0)
				return -(double)(tpDeprived+fpDeprived)/(double)(saPos+saNeg);
			else
				return 100*((double)(tpFavored+fpFavored)/(double)(nSaPos+nSaNeg)
						-(double)(tpDeprived+fpDeprived)/(double)(saPos+saNeg));
		}
	}
	public static double DiscriminationScore_WithoutSA(double[] predictions, int[] trueLabels){
		tpDeprived=0;tnDeprived=0;fnDeprived=0;fpDeprived=0;
		tpFavored=0;tnFavored=0;fnFavored=0;fpFavored=0;
		for (int i=0; i<windowSize; i++){
			String[] splits=windowList[i].instance.toString().split(",");
			if (splits[saIndex].equals(saVal)){ //Deprived
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
		if ((saPos+saNeg)==0)
			return 100*(double)(tpFavored+fpFavored)/(double)(nSaPos+nSaNeg);
		else{
			if ((nSaPos+nSaNeg)==0)
				return -(double)(tpDeprived+fpDeprived)/(double)(saPos+saNeg);
			else
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
	/* method to sort the 2-D arrays
	  * @param arrayToSort A 2-D array which we want to sort
	  * @param type 1 is descending order and type 2 is for ascending order
	  * @return sorted array
	  */
	public static double[][] sorting(double [][] arrayToSort,int length,int type){
		int max=length;
		double val1=0,val2=0;
		double [][]sortedArray=new double[length][2];
		double [][] temp=new double[1][2];
		for(int index=0;index<length;index++)
			for(int i=0;i<max-1;i++){
				try{
					val1=arrayToSort[i][1];
					val2=arrayToSort[i+1][1];

					if(val1<val2 && type==1){  //swapping for sort descending
						System.arraycopy(arrayToSort[i],0,temp[0],0,2);
						System.arraycopy(arrayToSort[i+1],0,arrayToSort[i],0,2);
						System.arraycopy(temp[0],0,arrayToSort[i+1],0,2);//System.out.println("val1 = "+val1+" new value of rec[] "+rec[i+1][20]+" i = "+i);
					}     //end of  if
					else if(val1>val2 && type==2){  //swapping for sort ascending
						System.arraycopy(arrayToSort[i],0,temp[0],0,2);
						System.arraycopy(arrayToSort[i+1],0,arrayToSort[i],0,2);
						System.arraycopy(temp[0],0,arrayToSort[i+1],0,2);//System.out.println("val1 = "+val1+" new value of rec[] "+rec[i+1][20]+" i = "+i);
					}     //end of else if

				} catch (NumberFormatException e){
					System.out.println(" Probelme with sorting during Massaging");
				}

			}//end of out for-i loop
		for(int i=0;i<length;i++)
			System.arraycopy(arrayToSort[i],0, sortedArray[i],0, 2);
		return sortedArray;
	}   // End of sorting function	 
}
