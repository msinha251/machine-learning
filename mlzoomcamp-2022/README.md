# mlzoomcamp-2022

This contains the course progression of <a href='https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/cohorts/2022'> mlzoomcamp-2022 </a> course.

### Projects 

- [Midterm Project](https://github.com/msinha251/mlzoomcamp-midterm-project-2022)


### Environment Setup on EC2:
* Created EC2 instance with Ubuntu image.<br>
* ssh into ec2 machine<br>
`ssh -i <>.pem ubuntu@<ec2-ip>` <br>
* update lib's <br>
`sudo apt-get update` <br>
* Install conda:<br>
Ref: <a href='https://www.how2shout.com/linux/how-to-install-anacondaon-ubuntu-22-04-lts-jammy/'> Article </a> <br>
**NOTE: Don't run the `Step3: Add Environment variable for Anaconda`**<br>
* Create conda virtual environment:<br>
`conda create -n ml-zoomcamp python=3.9`<br>
`conda install numpy pandas scikit-learn seaborn jupyter`
`pip install xgboost`
`pip install tensorflow`

* Open jupyter notebook
`jupyter notebook`

## Course Structure:

#### Week 1: Introduction to Machine Learning
#### Week 2: Machine learning for Regression
#### Week 3: Machine learning for Classification
#### Week 4: Evaluation Metrics for Classification
#### Week 5: Deploying Machine Learning Models
#### Week 6: Decision Trees and Ensemble Methods
#### Week 7: Production-Ready Machine Learning (Bento ML)
### [MID TERM PROJECT](https://github.com/msinha251/mlzoomcamp-midterm-project-2022)

#### Week 8: Neural Networks and Deep Learning




