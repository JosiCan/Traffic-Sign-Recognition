


all: RoadSigns-SVM.py
	python3 RoadSigns-SVM.py > output.txt

clean:
	rm output.txt
