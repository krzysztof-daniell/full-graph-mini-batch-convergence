void runContainer(String containerName, String imageName) {
    sh """docker run \
        -d \
        -it \
        --ipc host \
        --rm \
        --name ${containerName}
        ${imageName}"""
}

void cloneRepository(
    String containerName,
    String repositoryURL, 
    String machineUserCredentials
) {
    sh """docker exec ${containerName} -c bash
            sh 'git clone https://${machineUserCredentials}@${repositoryURL.minus("https://")}'
        """
}

void turnOffHyperthreading() {
    sh """docker exec 
        -w /full-graph-mini-batch-convergence/experiments ${containerName} \
        -c bash \
            'bash turn_off_hyperthreading.sh'
    """
}

String createExperiment(String imageName) {
    containerName = 'sigopt-create-experiment-${env.BUILD_NUMBER.padLeft(5, "0")}'

    runContainer(containerName, imageName)

    // TODO: Credentials
    withCredentials {
        cloneRepository(containerName, env.REPOSITORY_URL, machineUserCredentials)
    }

    script {
        experimentID = sh (
            returnStdout: true,
            script: """docker exec
                -w /full-graph-mini-batch-convergence/experiments ${containerName} \
                -c bash \
                    'python run.py --create-experiment'
            """
        ).trim()
    }

    sh 'docker stop ${containerName}'

    return experimentID
}

void runExperiment(
    String containerName,
    String experimentID,
    String experimentTarget
    String model,
    String dataset,
    String trainingMethod
) {
    sh """docker exec 
    -w /full-graph-mini-batch-convergence/experiments ${containerName} -c bash \
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
    agent { any }
    environment {
        REPOSITORY_URL = 'https://github.com/ksadowski13/full-graph-mini-batch-convergence.git'

        IMAGE_NAME = 'dgl-sigopt'

        NUMBER_OF_AWS_INSTANCES = 8
    }
    parameters {
        string(
            name: 'AWS_INSTANCES_IDS',
            description: 'AWS Instances IDs (format: id_1;id_2,...)'
        )
        string(
            name: 'AWS_INSTANCES_IPS',
            description: 'AWS Instances IPs (format: ip_1;ip_2,...)'
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
            choices: ['ogbn-arxiv', 'ogbn-mag', 'ogbn-products', 'ogbn-proteins']
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
                    if (!env.EXPERIMENT_ID.notBlank()) {
                        env.EXPERIMENT_ID = createExperiment(env.IMAGE_NAME)
                    }

                    sh 'echo "Experiment ID: ${env.EXPERIMENT_ID}"'

                    // TODO: Parse instances IDs and IPs int lists
                }
            }
        }
        stage('Run Experiment') {
            steps {
                parallel {
                    for (int i = 0; i < env.NUMBER_OF_AWS_INSTANCES; i++) {
                        stage('Instance: ${i}') {
                            String containerName = 'dgl-sigopt-optimization'

                            // TODO: Credentials
                            withCredentials {
                                // TODO: Start AWS Instance

                                // connect to AWS instance, 
                                // turn off hyperthreading (check if it works when run from docker),
                                // run docker container, 
                                // clone repo to container,
                                // run experiment in container
                                sh """ssh XXXXXXXXXXXXXXX \
                                    '${runContainer(containerName, env.IMAGE_NAME)} \
                                    && ${turnOffHyperthreading()} \
                                    && ${cloneRepository(containerName, env.REPOSITORY_URL, machineUserCredentials)} \
                                    && ${runExperiment(containerName, env.EXPERIMENT_ID, param.OPTIMIZATION_TARGET, param.MODEL, param.DATASET, param.TRAINING_METHOD)} \
                                    && docker stop ${containerName}'
                                """

                                // TODO: Stop AWS Instance
                            }
                        }
                    }
                }
            }
        }   
    }
}