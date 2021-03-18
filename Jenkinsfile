#!groovy

node('executor') {

    // HACK: fix permissions on the Neo4j conf + data volumes:
    stage("Clean") {
        sh "[ -d ./conf ] && sudo chmod -R 777 ./conf && sudo chown -R ubuntu:ubuntu ./conf || return 0"
        sh "[ -d ./data ] && sudo chmod -R 777 ./data && sudo chown -R ubuntu:ubuntu ./data || return 0"
    }

    checkout scm

    def authorName = sh(returnStdout: true, script: 'git --no-pager show --format="%an" --no-patch')
    def commitHash = sh(returnStdout: true, script: 'git rev-parse HEAD | cut -c-7').trim()
    def imageTag   = "${env.BUILD_NUMBER}-${commitHash}"

    def pennsieveNexusCreds = usernamePassword(
        credentialsId: "pennsieve-nexus-ci-login",
        usernameVariable: "PENNSIEVE_NEXUS_USER",
        passwordVariable: "PENNSIEVE_NEXUS_PW"
    )

    try {
        // stage("Typecheck & Lint") {
        //     withCredentials([pennsieveNexusCreds]) {
        //         try {
        //           sh "IMAGE_TAG=${imageTag} make lint-ci"
        //         } catch (e) {
        //           sh "echo 'Run \"make format\" to fix issues'"
        //           throw e
        //         }
        //     }
        // }
        stage("Test") {
            withCredentials([pennsieveNexusCreds]) {
                try {
                  sh "IMAGE_TAG=${imageTag} make test-ci"
                } finally {
                  sh "make docker-clean"
                }
            }
        }

        if (["master"].contains(env.BRANCH_NAME)) {
            stage("Docker") {

                sh "[ -d ./conf ] && sudo chmod -R 777 ./conf && sudo chown -R ubuntu:ubuntu ./conf || return 0"
                sh "[ -d ./data ] && sudo chmod -R 777 ./data && sudo chown -R ubuntu:ubuntu ./data || return 0"

                withCredentials([pennsieveNexusCreds]) {
                    sh "IMAGE_TAG=${imageTag} make containers"
                }

                sh "docker tag pennsieve/model-service:${imageTag} pennsieve/model-service:latest"
                sh "docker push pennsieve/model-service:latest"
                sh "docker push pennsieve/model-service:${imageTag}"

                sh "docker tag pennsieve/model-publish:${imageTag} pennsieve/model-publish:latest"
                sh "docker push pennsieve/model-publish:latest"
                sh "docker push pennsieve/model-publish:${imageTag}"
            }

            stage("Deploy") {
                parallel "model-service": {
                    build job: "service-deploy/pennsieve-non-prod/us-east-1/dev-vpc-use1/dev/model-service",
                        parameters: [
                        string(name: 'IMAGE_TAG', value: imageTag),
                        string(name: 'TERRAFORM_ACTION', value: 'apply')
                    ]
                },
                "model-publish": {
                    build job: "service-deploy/pennsieve-non-prod/us-east-1/dev-vpc-use1/dev/model-publish",
                        parameters: [
                        string(name: 'IMAGE_TAG', value: imageTag),
                        string(name: 'TERRAFORM_ACTION', value: 'apply')
                    ]
                }
            }
        }
    } catch (e) {
        slackSend(color: '#b20000', message: "FAILED: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]' (${env.BUILD_URL}) by ${authorName}")
        throw e
    }
    slackSend(color: '#006600', message: "SUCCESSFUL: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]' (${env.BUILD_URL}) by ${authorName}")
}
