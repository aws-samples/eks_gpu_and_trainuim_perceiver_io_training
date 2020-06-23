FROM python:3.6
USER root
RUN python -m pip install --upgrade pip
RUN pip install awscli
RUN pip install pyspark
RUN pip install findspark

RUN wget http://archive.apache.org/dist/spark/spark-2.4.6/spark-2.4.6-bin-hadoop2.7.tgz
RUN tar xvzf spark-2.4.6-bin-hadoop2.7.tgz
RUN mv spark-2.4.6-bin-hadoop2.7 /opt/spark-2.4.6
RUN ln -s /opt/spark-2.4.6 /opt/spark
RUN rm spark-2.4.6-bin-hadoop2.7.tgz
RUN export SPARK_HOME=/opt/spark
RUN export PATH=$SPARK_HOME/bin:$PATH
RUN apt-get update -y
RUN apt-get install -y openjdk-8-jre
RUN apt-get install -y gcc 
RUN sudo apt-get install -y  python3 python-dev python3-dev \
     build-essential libssl-dev libffi-dev \
     libxml2-dev libxslt1-dev zlib1g-dev \
     python-pip
