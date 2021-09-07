void runContainer(
    String containerName,
    String imageName
) {
    sh """docker run \
    -d \
    -it \
    --name ${containerName}
    ${imageName}"""
}

void orchestrateExperiment(
    String containerName,
    Boolean createCluster,
    String modelName,
    String dataset,
    String experimentTarget,
    String trainingMethod,
) {

    String experiment = "${dataset}_${experimentTarget}_${trainingMethod}"

    if (createCluster) {
        sh """docker exec \
        -w /full-graph-mini-batch-covergence/experiments/${modelName} \
        ${containerName} -c bash \
            'sigopt cluster create -f cluster cluster_${experiment}.yml'
        """
    }

    sh """docker exec \
    -w /full-graph-mini-batch-covergence/experiments/${modelName} \
    ${containerName} -c bash \
        'sigopt cluster run -f cluster run_${experiment}.yml \
        && sigopt cluster optimize -e experiment_${experiment}.yml'
    """
}

void orchestrateExperiments(
    String containerName,
    Boolean createCluster,
    Boolean runGATArxivAccuracyMiniBatch,
    Boolean runGATArxivAccuracyFullGraph,
    Boolean runGATArxivSpeedMiniBatch,
    Boolean runGATArxivSpeedFullGraph,
    Boolean runGATProductsAccuracyMiniBatch,
    Boolean runGATProductsAccuracyFullGraph,
    Boolean runGATProductsSpeedMiniBatch,
    Boolean runGATProductsSpeedFullGraph,
    Boolean runGATProteinsAccuracyMiniBatch,
    Boolean runGATProteinsAccuracyFullGraph,
    Boolean runGATProteinsSpeedMiniBatch,
    Boolean runGATProteinsSpeedFullGraph,
    Boolean runGraphSAGEArxivAccuracyMiniBatch,
    Boolean runGraphSAGEArxivAccuracyFullGraph,
    Boolean runGraphSAGEArxivSpeedMiniBatch,
    Boolean runGraphSAGEArxivSpeedFullGraph,
    Boolean runGraphSAGEProductsAccuracyMiniBatch,
    Boolean runGraphSAGEProductsAccuracyFullGraph,
    Boolean runGraphSAGEProductsSpeedMiniBatch,
    Boolean runGraphSAGEProductsSpeedFullGraph,
    Boolean runGraphSAGEProteinsAccuracyMiniBatch,
    Boolean runGraphSAGEProteinsAccuracyFullGraph,
    Boolean runGraphSAGEProteinsSpeedMiniBatch,
    Boolean runGraphSAGEProteinsSpeedFullGraph,
    Boolean runRGCNMagAccuracyMiniBatch,
    Boolean runRGCNMagAccuracyFullGraph,
    Boolean runRGCNMagSpeedMiniBatch,
    Boolean runRGCNMagSpeedFullGraph
) {
    // :::: GAT ::::
    // -- ogbn-arxiv --
    if (runGATArxivAccuracyMiniBatch) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "gat",
            "ogbn_arxiv",
            "accuracy",
            "mini_batch"
        )
    }

    if (runGATArxivAccuracyFullGraph) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "gat",
            "ogbn_arxiv",
            "accuracy",
            "full_graph"
        )
    }

    if (runGATArxivSpeedMiniBatch) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "gat",
            "ogbn_arxiv",
            "speed",
            "mini_batch"
        )
    }

    if (runGATArxivSpeedFullGraph) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "gat",
            "ogbn_arxiv",
            "speed",
            "full_graph"
        )
    }

    if (runGATArxivSpeedFullGraph) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "gat",
            "ogbn_arxiv",
            "speed",
            "full_graph"
        )
    }

    // -- ogbn-products --
    if (runGATProductsAccuracyMiniBatch) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "gat",
            "ogbn_products",
            "accuracy",
            "mini_batch"
        )
    }

    if (runGATProductsAccuracyFullGraph) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "gat",
            "ogbn_products",
            "accuracy",
            "full_graph"
        )
    }
    
    if (runGATProductsSpeedMiniBatch) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "gat",
            "ogbn_products",
            "speed",
            "mini_batch"
        )
    }

    if (runGATProductsSpeedFullGraph) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "gat",
            "ogbn_products",
            "speed",
            "full_graph"
        )
    }

    // -- ogbn-proteins --
    if (runGATProteinsAccuracyMiniBatch) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "gat",
            "ogbn_proteins",
            "accuracy",
            "mini_batch"
        )
    }

    if (runGATProteinsAccuracyFullGraph) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "gat",
            "ogbn_proteins",
            "accuracy",
            "full_graph"
        )
    }
    
    if (runGATProteinsSpeedMiniBatch) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "gat",
            "ogbn_proteins",
            "speed",
            "mini_batch"
        )
    }

    if (runGATProteinsSpeedFullGraph) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "gat",
            "ogbn_proteins",
            "speed",
            "full_graph"
        )
    }

    // :::: GraphSAGE ::::
    // -- ogbn-arxiv --
    if (runGraphSAGEArxivAccuracyMiniBatch) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "graphsage",
            "ogbn_arxiv",
            "accuracy",
            "mini_batch"
        )
    }

    if (runGraphSAGEArxivAccuracyFullGraph) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "graphsage",
            "ogbn_arxiv",
            "accuracy",
            "full_graph"
        )
    }

    if (runGraphSAGEArxivSpeedMiniBatch) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "graphsage",
            "ogbn_arxiv",
            "speed",
            "mini_batch"
        )
    }

    if (runGraphSAGEArxivSpeedFullGraph) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "graphsage",
            "ogbn_arxiv",
            "speed",
            "full_graph"
        )
    }

    if (runGraphSAGEArxivSpeedFullGraph) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "graphsage",
            "ogbn_arxiv",
            "speed",
            "full_graph"
        )
    }

    // -- ogbn-products --
    if (runGraphSAGEProductsAccuracyMiniBatch) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "graphsage",
            "ogbn_products",
            "accuracy",
            "mini_batch"
        )
    }

    if (runGraphSAGEProductsAccuracyFullGraph) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "graphsage",
            "ogbn_products",
            "accuracy",
            "full_graph"
        )
    }
    
    if (runGraphSAGEProductsSpeedMiniBatch) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "graphsage",
            "ogbn_products",
            "speed",
            "mini_batch"
        )
    }

    if (runGraphSAGEProductsSpeedFullGraph) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "graphsage",
            "ogbn_products",
            "speed",
            "full_graph"
        )
    }

    // -- ogbn-proteins --
    if (runGraphSAGEProteinsAccuracyMiniBatch) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "graphsage",
            "ogbn_proteins",
            "accuracy",
            "mini_batch"
        )
    }

    if (runGraphSAGEProteinsAccuracyFullGraph) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "graphsage",
            "ogbn_proteins",
            "accuracy",
            "full_graph"
        )
    }
    
    if (runGraphSAGEProteinsSpeedMiniBatch) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "graphsage",
            "ogbn_proteins",
            "speed",
            "mini_batch"
        )
    }

    if (runGraphSAGEProteinsSpeedFullGraph) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "graphsage",
            "ogbn_proteins",
            "speed",
            "full_graph"
        )
    }

    // :::: RGCN ::::
    // -- ogbn-mag --
    if (runGraphSAGEArxivAccuracyMiniBatch) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "rgcn",
            "ogbn_mag",
            "accuracy",
            "mini_batch"
        )
    }

    if (runGraphSAGEArxivAccuracyFullGraph) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "rgcn",
            "ogbn_mag",
            "accuracy",
            "full_graph"
        )
    }

    if (runGraphSAGEArxivSpeedMiniBatch) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "rgcn",
            "ogbn_mag",
            "speed",
            "mini_batch"
        )
    }

    if (runGraphSAGEArxivSpeedFullGraph) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "rgcn",
            "ogbn_mag",
            "speed",
            "full_graph"
        )
    }

    if (runGraphSAGEArxivSpeedFullGraph) {
        orchestrateExperiment(
            containerName,
            createCluster,
            "rgcn",
            "ogbn_mag",
            "speed",
            "full_graph"
        )
    }
}

pipeline {
    agent {}
    environment {
        REPOSITORY_URL = "https://github.com/ksadowski13/full-graph-mini-batch-convergence.git"
        MACHINE_CREDENTIALS = credentials("dgl-intel-github")

        IMAGE_NAME = "sigopt-ci-orchestration"
    }
    parameters {
        booleanParam(
            name: "CREATE_CLUSTER",
            defaultValue: false,
            description: "Whethever to create AWS cluster for the experiments."
        )

        booleanParam(
            name: "RUN_GAT_ARXIV_ACCURACY_MINI_BATCH",
            defaultValue: false,
            description: "- GAT - ogbn-arxiv - accuracy - mini-batch -"
        )
        booleanParam(
            name: "RUN_GAT_ARXIV_ACCURACY_FULL_GRAPH",
            defaultValue: false,
            description: "- GAT - ogbn-arxiv - accuracy - full-graph -"
        )
        booleanParam(
            name: "RUN_GAT_ARXIV_SPEED_MINI_BATCH",
            defaultValue: false,
            description: "- GAT - ogbn-arxiv - speed - mini-batch -"
        )
        booleanParam(
            name: "RUN_GAT_ARXIV_SPEED_FULL_GRAPH",
            defaultValue: false,
            description: "- GAT - ogbn-arxiv - speed - full-graph -"
        )
        booleanParam(
            name: "RUN_GAT_PRODUCTS_ACCURACY_MINI_BATCH",
            defaultValue: false,
            description: "- GAT - ogbn-products - accuracy - mini-batch -"
        )
        booleanParam(
            name: "RUN_GAT_PRODUCTS_ACCURACY_FULL_GRAPH",
            defaultValue: false,
            description: "- GAT - ogbn-products - accuracy - full-graph -"
        )
        booleanParam(
            name: "RUN_GAT_PRODUCTS_SPEED_MINI_BATCH",
            defaultValue: false,
            description: "- GAT - ogbn-products - speed - mini-batch -"
        )
        booleanParam(
            name: "RUN_GAT_PRODUCTS_SPEED_FULL_GRAPH",
            defaultValue: false,
            description: "- GAT - ogbn-products - speed - full-graph -"
        )
        booleanParam(
            name: "RUN_GAT_PROTEINS_ACCURACY_MINI_BATCH",
            defaultValue: false,
            description: "- GAT - ogbn-proteins - accuracy - mini-batch -"
        )
        booleanParam(
            name: "RUN_GAT_PROTEINS_ACCURACY_FULL_GRAPH",
            defaultValue: false,
            description: "- GAT - ogbn-proteins - accuracy - full-graph -"
        )
        booleanParam(
            name: "RUN_GAT_PROTEINS_SPEED_MINI_BATCH",
            defaultValue: false,
            description: "- GAT - ogbn-proteins - speed - mini-batch -"
        )
        booleanParam(
            name: "RUN_GAT_PROTEINS_SPEED_FULL_GRAPH",
            defaultValue: false,
            description: "- GAT - ogbn-proteins - speed - full-graph -"
        )

        booleanParam(
            name: "RUN_GRAPHSAGE_ARXIV_ACCURACY_MINI_BATCH",
            defaultValue: false,
            description: "- GraphSAGE - ogbn-arxiv - accuracy - mini-batch -"
        )
        booleanParam(
            name: "RUN_GRAPHSAGE_ARXIV_ACCURACY_FULL_GRAPH",
            defaultValue: false,
            description: "- GraphSAGE - ogbn-arxiv - accuracy - full-graph -"
        )
        booleanParam(
            name: "RUN_GRAPHSAGE_ARXIV_SPEED_MINI_BATCH",
            defaultValue: false,
            description: "- GraphSAGE - ogbn-arxiv - speed - mini-batch -"
        )
        booleanParam(
            name: "RUN_GRAPHSAGE_ARXIV_SPEED_FULL_GRAPH",
            defaultValue: false,
            description: "- GraphSAGE - ogbn-arxiv - speed - full-graph -"
        )
        booleanParam(
            name: "RUN_GRAPHSAGE_PRODUCTS_ACCURACY_MINI_BATCH",
            defaultValue: false,
            description: "- GraphSAGE - ogbn-products - accuracy - mini-batch -"
        )
        booleanParam(
            name: "RUN_GRAPHSAGE_PRODUCTS_ACCURACY_FULL_GRAPH",
            defaultValue: false,
            description: "- GraphSAGE - ogbn-products - accuracy - full-graph -"
        )
        booleanParam(
            name: "RUN_GRAPHSAGE_PRODUCTS_SPEED_MINI_BATCH",
            defaultValue: false,
            description: "- GraphSAGE - ogbn-products - speed - mini-batch -"
        )
        booleanParam(
            name: "RUN_GRAPHSAGE_PRODUCTS_SPEED_FULL_GRAPH",
            defaultValue: false,
            description: "- GraphSAGE - ogbn-products - speed - full-graph -"
        )
        booleanParam(
            name: "RUN_GRAPHSAGE_PROTEINS_ACCURACY_MINI_BATCH",
            defaultValue: false,
            description: "- GraphSAGE - ogbn-proteins - accuracy - mini-batch -"
        )
        booleanParam(
            name: "RUN_GRAPHSAGE_PROTEINS_ACCURACY_FULL_GRAPH",
            defaultValue: false,
            description: '- GraphSAGE - ogbn-proteins - accuracy - full-graph -'
        )
        booleanParam(
            name: "RUN_GRAPHSAGE_PROTEINS_SPEED_MINI_BATCH",
            defaultValue: false,
            description: '- GraphSAGE - ogbn-proteins - speed - mini-batch -'
        )
        booleanParam(
            name: "RUN_GRAPHSAGE_PROTEINS_SPEED_FULL_GRAPH",
            defaultValue: false,
            description: '- GraphSAGE - ogbn-proteins - speed - full-graph -'
        )

        booleanParam(
            name: "RUN_RGCN_MAG_ACCURACY_MINI_BATCH",
            defaultValue: false,
            description: '- RGCN - ogbn-mag - accuracy - mini-batch -'
        )
        booleanParam(
            name: "RUN_RGCN_MAG_ACCURACY_FULL_GRAPH",
            defaultValue: false,
            description: '- RGCN - ogbn-mag - accuracy - full-graph -'
        )
        booleanParam(
            name: "RUN_RGCN_MAG_SPEED_MINI_BATCH",
            defaultValue: false,
            description: '- RGCN - ogbn-mag - speed - mini-batch -'
        )
        booleanParam(
            name: "RUN_RGCN_MAG_SPEED_FULL_GRAPH",
            defaultValue: false,
            description: '- RGCN - ogbn-mag - speed - full-graph -'
        )
    }
    stages {
        stage("Run Container") {
            steps {
                env.CONTAINER_NAME = "sigopt-experiment-orchestration-${env.BUILD_NUMBER.padLeft(5, '0')}"

                runContainer(env.CONTAINER_NAME, env.IMAGE_NAME)
            }
        }
        stage("Clone Repository") {
            steps {
                sh """docker exec ${env.CONTAINER_NAME} bash -c \
                    'git clone https://${env.MACHINE_CREDENTIALS}@${env.REPOSITORY_URL.minus("https://")}'
                """
            }
        }
        stage("Orchestrate Experiments") {
            orchestrateExperiments(
                env.containerName,
                params.createCluster,
                params.runGATArxivAccuracyMiniBatch,
                params.runGATArxivAccuracyFullGraph,
                params.runGATArxivSpeedMiniBatch,
                params.runGATArxivSpeedFullGraph,
                params.runGATProductsAccuracyMiniBatch,
                params.runGATProductsAccuracyFullGraph,
                params.runGATProductsSpeedMiniBatch,
                params.runGATProductsSpeedFullGraph,
                params.runGATProteinsAccuracyMiniBatch,
                params.runGATProteinsAccuracyFullGraph,
                params.runGATProteinsSpeedMiniBatch,
                params.runGATProteinsSpeedFullGraph,
                params.runGraphSAGEArxivAccuracyMiniBatch,
                params.runGraphSAGEArxivAccuracyFullGraph,
                params.runGraphSAGEArxivSpeedMiniBatch,
                params.runGraphSAGEArxivSpeedFullGraph,
                params.runGraphSAGEProductsAccuracyMiniBatch,
                params.runGraphSAGEProductsAccuracyFullGraph,
                params.runGraphSAGEProductsSpeedMiniBatch,
                params.runGraphSAGEProductsSpeedFullGraph,
                params.runGraphSAGEProteinsAccuracyMiniBatch,
                params.runGraphSAGEProteinsAccuracyFullGraph,
                params.runGraphSAGEProteinsSpeedMiniBatch,
                params.runGraphSAGEProteinsSpeedFullGraph,
                params.runRGCNMagAccuracyMiniBatch,
                params.runRGCNMagAccuracyFullGraph,
                params.runRGCNMagSpeedMiniBatch,
                params.runRGCNMagSpeedFullGraph
            )
        }
    }
}