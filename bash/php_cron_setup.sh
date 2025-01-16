# mb
#!/bin/bash

# Define the remote PHP backup script URL
BACKUP_URL=${1:-"http://httpbin.org/status/200"}

# Define the log file location
LOG_FILE="./mb_backup_php.log"

# Define the email address to notify in case of failure
EMAIL_ADDRESS="your-email@example.com"

# Function to log a message
log_message() {
    date +"%Y-%m-%d %H:%M:%S" >> "$LOG_FILE"
    echo "$@" >> "$LOG_FILE"
}

# Function to send an email notification
send_email() {
    subject="Backup Script Failure"
    message="The backup script failed to run successfully. Please check the log file for more details."
    mail -s "$subject" "$EMAIL_ADDRESS" <<< "$message"
}

# Run the PHP backup script using curl
log_message "Running backup script at $BACKUP_URL"

# Attempt to run the script
result=$(curl -s "$BACKUP_URL")

# Check if the response is empty or contains an error message
if [ -z "$result" ] || [[ "$result" =~ "error" ]]; then
    log_message "Backup script failed. Received empty reply or error message: $result"
    # send_email
    exit 1
else
    log_message "Backup script completed successfully."
    exit 0
fi