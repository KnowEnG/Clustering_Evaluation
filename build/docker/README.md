# Building The Clustering Evaluation Pipeline Docker Image
The Dockefile in this directory contains all the commands, in order, needed to build the **Clustering Evaluation** docker image.


* Run the "make" command to build the **Clustering Evaluation** docker image (output: docker image called "clustering_evaluation" and a tag with today's date and time):
```
    make build_docker_image 
```

* Login to docker hub. When prompted, enter your password and press enter:
```
    make login_to_dockerhub username=(enter your docker login here) email=(enter your email here)
```

* Upload your image to docker hub:
```
    make push_to_dockerhub
```

* * * 
## How to run this docker image
* * * 

### * Run the following command with the specified docker image:
```
docker run -v \`pwd\`:/home/test/run_dir/ -it knowengdev/clustering_evaluation:{tag}
```

### * Change directory to the "test" directory
```
cd test
```

### * Create the local directory "run_dir" and place all the run files in it
```
make env_setup
```

### * Run the GeneSet Characterization Pipeline
```
make run_cluster_eval
```
