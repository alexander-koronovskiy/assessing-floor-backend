pipeline {

  agent any

  environment {
    CI = 'true'
    IMAGENAME = '/yks/yks-inner/assessing-floor'
    REGISTRY = 'registry.gitlab.com'
    registryCredential = 'gitlab-user'
  }

  parameters {
    string(name: 'CD_PORT_PRODUCTION', defaultValue: '5500', description: 'Which port to choose for PRODUCTION?')
    string(name: 'PRODUCTION_BRANCH', defaultValue: 'main', description: 'What branch in PRODUCTION?')
    string(name: 'AGENT_LABEL', defaultValue: 'yks-develop', description: 'On which server?')
  }


  options {
    buildDiscarder(logRotator(numToKeepStr: '2', artifactNumToKeepStr: '2'))
  }

  stages {

    stage('Debug') {
      steps {
        sh 'printenv'
        sh 'node -v || true'
        sh 'docker -v || true'
        sh 'docker info || true'
        sh 'docker-compose -v || true'
      }
    }

    stage('Build image & push in registry') {
      when {
        expression {
          GIT_BRANCH == 'main' || GIT_BRANCH == params.PRODUCTION_BRANCH
        }
      }
      steps {
        script {
          docker.withRegistry('https://' + REGISTRY + IMAGENAME, registryCredential) {

            def customImage = docker.build("${REGISTRY}${IMAGENAME}/${GIT_BRANCH}:${BUILD_NUMBER}")

            customImage.push()
            customImage.push('latest')

          }
        }
      }
      post {
        cleanup {
          sh 'docker rmi ${REGISTRY}${IMAGENAME}/${GIT_BRANCH}:${BUILD_NUMBER} || true' // untaget
          sh 'docker rmi ${REGISTRY}${IMAGENAME}/${GIT_BRANCH}:latest || true' // rmi
          sh 'docker rmi $(docker images -f "dangling=true" -q) || true' // remove all dangling=true
        }
        success {
          sendTelegram(" ${REGISTRY}${IMAGENAME}/${GIT_BRANCH}:${BUILD_NUMBER} :: ${currentBuild.currentResult}")
        }
        unstable {
          sendTelegram(" ${REGISTRY}${IMAGENAME}/${GIT_BRANCH}:${BUILD_NUMBER} :: ${currentBuild.currentResult}")
        }
        failure {
          sendTelegram(currentBuild.currentResult)
        }
      }
    }

    stage('Deploy') {
      when {
        expression {
          GIT_BRANCH == 'main' || GIT_BRANCH == params.PRODUCTION_BRANCH
        }
      }
      environment {
        CD_PORT = selectPort(GIT_BRANCH)
      }
      agent {
        label params.AGENT_LABEL
      }
      steps {
        script {
          docker.withRegistry('https://' + REGISTRY + IMAGENAME, registryCredential) {
            sh('docker-compose pull')
            sh('docker-compose -f docker-compose.yml up -d --force-recreate')
            sh('docker ps')
          }
        }
      }
      post {
        cleanup {
          sh 'docker rmi $(docker images -f "dangling=true" -q) || true'
        }
        success {
          sendTelegram(currentBuild.currentResult)
        }
        unstable {
          sendTelegram(currentBuild.currentResult)
        }
        failure {
          sendTelegram(currentBuild.currentResult)
        }
      }
    }

  }
}


// Отправляет сообщение в телеграм чат (бот)
Object sendTelegram (String message) {
  str = """
    curl --silent --location --request POST 'https://tg-bot.ykdev.ru/send' \
    --header 'access-token: CluN8RoMrvFNSqxX7pl7nifRQty42zus' \
    --header 'Content-Type: application/json' \
    --data-raw '{
        "message": "${BUILD_DISPLAY_NAME} :: ${GIT_URL} :: ${JOB_BASE_NAME} :: ${STAGE_NAME} :: ${message}"
    }'
  """
  return sh(
    script: str,
    returnStdout: true
  )
}

// Утанавливает порт в зависимости от условий
Object selectPort(String branch) {
  if (branch == params.PRODUCTION_BRANCH) {
    return params.CD_PORT_PRODUCTION
  }
  return params.CD_PORT_DEVELOP
}
