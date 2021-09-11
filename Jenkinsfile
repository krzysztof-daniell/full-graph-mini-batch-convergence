void startAWSInstances(String instanceIDs) {
    sh 'aws ec2 start-instances --instance-ids ${instanceIDs}'
}

void stopAWSInstances(String instanceIDs) {
    sh 'aws ec2 stop-instances --instance-ids ${instanceIDs}'
}

void runContainer(String containerName, String imageName) {
    sh """docker run \
        -d \
        -it \
        --ipc host \
        --rm \
        --name ${containerName}
        ${imageName}
    """
}

void turnOffHyperthreading() {
    sh """docker exec \
        -w /full-graph-mini-batch-convergence/experiments ${containerName} \
        -c bash \
            'bash turn_off_hyperthreading.sh'
    """
}

void cloneRepository(
    String containerName,
    String repositoryURL,
    String machineUserCredentials
) {
    sh """docker exec ${containerName} -c bash
        'git clone https://${machineUserCredentials}@${repositoryURL.minus("https://")}'
    """
}

String createExperiment(String imageName, String machineUserCredentials) {
    script {
        containerName = 'sigopt-create-experiment-${env.BUILD_NUMBER.padLeft(5, "0")}'

        runContainer(containerName, imageName)
        cloneRepository(containerName, env.REPOSITORY_URL, machineUserCredentials)

        String experimentID = sh (
            returnStdout: true,
            script: """docker exec
                -w /full-graph-mini-batch-convergence/experiments ${containerName} \
                -c bash \
                    'python run.py --create-experiment'
            """
        ).trim()
    }

    sh 'docker stop ${containerName}'
    sh 'echo "Experiment ID: ${experimentID}"'

    return experimentID
}

void runExperiment(
    String containerName,
    String experimentID,
    String experimentTarget,
    String model,
    String dataset,
    String trainingMethod
) {
    sh """docker exec \
        -w /full-graph-mini-batch-convergence/experiments ${containerName} \
        -c bash \
            'python run.py \
                --optimize \
                --experiment-id ${experimentID} \
                --experiment-target ${experimentTarget} \
                --model ${model} \
                --dataset ${dataset} \
                --training-method ${trainingMethod}'
    """
}

pipeline {
    agent any
    environment {
        REPOSITORY_URL = 'https://github.com/ksadowski13/full-graph-mini-batch-convergence.git'

        IMAGE_NAME = 'dgl-sigopt'

        AWS_DEFAULT_REGION='us-west-1'
    }
    parameters {
        string(
            name: 'AWS_INSTANCE_IDS',
            description: 'AWS Instances IDs (format: id_1 id_2 ...)'
        )
        string(
            name: 'AWS_INSTANCE_IPS',
            description: 'AWS Instances IPs (format: ip_1 ip_2 ...)'
        )

        string(
            name: 'EXPERIMENT_ID',
            description: 'Experiment ID (if not provided, new experiment will be created)'
        )

        choice(
            name: 'OPTIMIZATION_TARGET',
            choices: ['accuracy', 'speed'],
            description: 'Optimization Target'
        )
        choice(
            name: 'MODEL',
            choices: ['gat', 'graphsage', 'rgcn'],
            description: "Model"
        )
        choice(
            name: 'DATASET',
            choices: ['ogbn-arxiv', 'ogbn-mag', 'ogbn-products', 'ogbn-proteins'],
            description: "Dataset"
        )
        choice(
            name: 'TRAINING_METHOD',
            choices: ['mini-batch', 'full-graph'],
            description: 'Training Method'
        )
    }
    stages {
        stage('Setup Experiment') {
            steps {
                script {
                    if (!env.EXPERIMENT_ID) {
                        withCredentials([
                            usernameColonPassword(
                                credentialsId: 'machine-user-credentials', 
                                variable: 'machineUserCredentials'
                            )
                        ]) {
                            env.EXPERIMENT_ID = createExperiment(
                                env.IMAGE_NAME, machineUserCredentials)
                        }
                    }
                }
            }
        }
        stage('Start AWS Instaces') {
            steps {
                withAWS(credentials: 'aws-credentials') {
                    startAWSInstances(param.AWS_INSTANCE_IDS)
                }
            }
        }
        stage('Orchestrate Experiment') {
            steps {
                script{
                    String containerName = 'dgl-sigopt-optimization-${env.BUILD_NUMBER.padLeft(5, "0")}'

                    parallel {
                        for (id in env.AWS_INSTANCE_IDS.split()) {
                            stage('Run Experiment on Instance: ${id}') {

                                // TODO: AWS Credentials
                                // TODO: Connect by AWS CLI
                                // TODO: Retry experiment on instance if fails
                                // TODO: Stop containers in post actions
                                // TODO: Check if turning off HT works from Docker
                                withCredentials([
                                    usernameColonPassword(
                                        credentialsId: 'machine-user-credentials', 
                                        variable: 'machineUserCredentials'
                                    )
                                ]) {
                                sh """ssh XXXXXXXXXXXXXXX \
                                        '${runContainer(
                                            containerName, env.IMAGE_NAME)} \
                                        && ${turnOffHyperthreading()} \
                                        && ${cloneRepository(
                                                containerName, 
                                                env.REPOSITORY_URL, 
                                                machineUserCredentials
                                            )} \
                                        && ${runExperiment(
                                                containerName, 
                                                env.EXPERIMENT_ID, 
                                                param.OPTIMIZATION_TARGET, 
                                                param.MODEL, 
                                                param.DATASET, 
                                                param.TRAINING_METHOD
                                            )} \
                                        && docker stop ${containerName}'
                                    """ 
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    post {
        always {
            withAWS(credentials: 'aws-credentials') {
                stopAWSInstances(param.AWS_INSTANCE_IDS)
            }
        }
    }
}
