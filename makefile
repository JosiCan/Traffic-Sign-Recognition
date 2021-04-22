


all: RoadSigns-SVM.py
	python3 RoadSigns-SVM.py > output.txt

tests: RoadSigns-Tests.py
	python3 RoadSigns-Tests.Py > output-tests.txt

clean:
	rm output.txt
