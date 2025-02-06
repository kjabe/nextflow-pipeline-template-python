#!/usr/bin/env nextflow

nextflow.enable.dsl=2

process load_data {
    conda 'env.yaml'
    debug true
    
    input:
    path id_path
    path data1
    path data2
    
    output:
    path "all_dataset.pkl"
    
    script:
    """
    echo "Current directory: \$(pwd)"
    echo "Contents of current directory: \$(ls -la)"
    echo "Input files:"
    echo "ID path: ${id_path}"
    echo "data1 path: ${data1}"
    echo "data2 path: ${data2}"
    
    python "${projectDir}/bin/load_data.py" "${id_path}" "${data1}" "${data2}" "${params.outdir}"
    """
}

process run_umap {
    conda 'env.yaml'
    
    input:
    path dataset
    
    output:
    path "umap_batch.pdf"
    path "umap_group.pdf"
    
    script:
    """
    python "${projectDir}/bin/run_umap.py" "${dataset}" "${params.outdir}"

    cp "${params.outdir}/umap_batch.pdf" .
    cp "${params.outdir}/umap_group.pdf" .
    """
}

process elastic_net {
    conda 'env.yaml'
    
    input:
    path dataset
    
    output:
    path "final_model.joblib"
    path "auc_value.joblib"
    path "model_coefficients.joblib"
    path "roc_curve.pdf"
    path "coefficients_plot.pdf"
    path "train_batch_distribution.pdf"
    path "test_batch_distribution.pdf"
    
    script:
    """
    python "${projectDir}/bin/elastic_net.py" "${dataset}" "${params.outdir}"

    cp "${params.outdir}/final_model.joblib" .
    cp "${params.outdir}/auc_value.joblib" .
    cp "${params.outdir}/model_coefficients.joblib" .
    cp "${params.outdir}/roc_curve.pdf" .
    cp "${params.outdir}/coefficients_plot.pdf" .
    cp "${params.outdir}/train_batch_distribution.pdf" .
    cp "${params.outdir}/test_batch_distribution.pdf" .
    """
}

workflow {
    log.info "Project Directory: $projectDir"
    log.info "Working Directory: $workDir"
    
    // Print the existence of key files
    log.info "Checking input files:"
    log.info "ID file exists: ${file(params.id_path).exists()}"
    log.info "data1 file exists: ${file(params.data1_path).exists()}"
    log.info "data2 file exists: ${file(params.data2_path).exists()}"
    
    // Create channels from input files
    id_path = Channel.fromPath(params.id_path)
    data1_path = Channel.fromPath(params.data1_path)
    data2_path = Channel.fromPath(params.data2_path)
    
    // Run processes
    dataset = load_data(id_path, data1_path, data2_path)
    umap_results = run_umap(dataset)
    elastic_net_results = elastic_net(dataset)
}