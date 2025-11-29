pipeline {
    agent any
    stages {
        stage('Environment Setup') {
            steps {
                echo 'Setting up environment...'
                // Check python version
                sh 'python3 --version'
                
                // Create virtual environment (optional but recommended in Jenkins)
                sh 'python3 -m venv venv'
            }
        }
        stage('Install Dependencies') {
            steps {
                echo 'Installing dependencies...'
                // Activate venv and install requirements
                sh '. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt && pip install flake8'
            }
        }
        stage('Linting') {
            steps {
                echo 'Running Linting...'
                // Run flake8
                sh '. venv/bin/activate && flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=venv'
            }
        }
        stage('Pipeline Verification') {
            steps {
                echo 'Running MLflow Pipeline...'
                // Run the pipeline script
                sh '. venv/bin/activate && python3 pipeline.py'
            }
        }
    }
    
    post {
        always {
            echo 'Cleaning up...'
            // Optional cleanup
            sh 'rm -rf venv'
        }
    }
}