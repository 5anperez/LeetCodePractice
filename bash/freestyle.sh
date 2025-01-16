# Prompt:
# i need a Bash script that sets up a cron job to run a PHP backup script hosted on another server. The script should use curl and handle "empty reply from server" errors gracefully. make sure to include logging and email notifications for failures.

# NOTES: 
# - Remember that $1 represents the first CL variable, which comes directly after the script name, e.g., bash setup_cron.sh --dry-run, where $1 == --dry-run







# gpt

#!/bin/bash

# Variables
LOG_FILE="/var/log/remote_backup.log"
EMAIL="your_email@example.com"
BACKUP_URL="http://example.com/backup.php"

# Function to log messages
log_message() {
    local message="$1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $message" | tee -a "$LOG_FILE"
}

# Function to send email notifications
send_email() {
    local subject="$1"
    local body="$2"
    echo "$body" | mail -s "$subject" "$EMAIL"
}

# Main backup process
log_message "Starting backup process."

# -s is silent mode, --retry 3 retries the request up to 3 times due to quick errors, --retry-delay 5: Adds a 5-second delay between retry attempts, --fail: Makes curl return a non-zero exit code if the HTTP response code is 400 or higher (e.g., 404 Not Found, 500 Internal Server Error). Without this, curl would consider such responses successful

# "$BACKUP_URL" is the URL of the PHP backup script you want to call, 2>&1: Redirects stderr (error messages) to stdout, so both standard output and error messages are captured together.

# $(...): The dollar sign w/ parenthesis right after is "command substitution". It captures the output of the curl command (both stdout and stderr) into the variable response.
response=$(curl -s --retry 3 --retry-delay 5 --fail "$BACKUP_URL" 2>&1)

# This line retrieves the exit code of the most recently executed command (in this case, curl). $? is a special shell variable that contains the exit status of the last executed command.
status=$?

if [[ $status -eq 0 ]]; then
    log_message "Backup completed successfully."
else
    error_message="Backup failed. Curl exited with status $status. Error: $response"
    log_message "$error_message"

    # Check for specific "empty reply from server" error
    if echo "$response" | grep -q "Empty reply from server"; then
        error_message+="\nEmpty reply from server encountered."
    fi

    # Send email notification
    send_email "Backup Failed" "$error_message"
fi

log_message "Backup process completed."

# Set up the cron job
# Add the following line to your crontab (use `crontab -e` to edit):
# 0 2 * * * /path/to/this_script.sh






















# ma
#!/bin/bash

# Define variables
# BACKUP_URL="http://your-remote-server.com/backup_script.php"

# Modify the backup url to accept a mock endpoint. 
# It uses :-, which is the default value operator in Bash,
# to send a request to the default http endpoint. The 
# default endpoint will be triggered if an argument is 
# not passed in, i.e, if the 1st positional arg is null.
BACKUP_URL=${1:-"http://httpbin.org/status/200"}

# LOG_FILE="/var/log/backup_server_script.log"
LOG_FILE="./backup_php.log"
EMAIL_ADDRESS="your-email@example.com"
SUBJECT="Backup Script Status"

# Function to log a message
function log_message {
    date +"%Y-%m-%d %H:%M:%S" >> "$LOG_FILE"
    echo "$*" >> "$LOG_FILE"
}

# Function to send an email
function send_email {
    echo -e "From: Backup Script <backup@example.com>\nTo: $EMAIL_ADDRESS\nSubject: $SUBJECT\n\n$1" | /usr/sbin/sendmail -t
}

# Function to run the backup script and handle errors
function run_backup {
    log_message "Running backup script..."

    # -S: Shows error messages only if an error occurs.
    # %{http_code} is a built-in feature of curl that extracts the HTTP response status code. When combined with the -w option, the write out option (--write-out), it allows the script to capture and use the status code for logging or error handling.
    # The -w option in curl lets you define a custom output format. You can include specific placeholders (variables that are enclosed in %{} and represent various metadata provided by curl) to extract details about the request or response.
    # E.g., if the server returns 200 OK, %{http_code} evaluates to 200. 
    # | xargs: Removes any unintended whitespace in the constructed string.
    CURL_OPTS=$(echo "-s -S -w %{http_code}" | xargs)
    RESPONSE_CODE=$(curl "$CURL_OPTS" "$BACKUP_URL")

    log_message "Backup script returned HTTP code: $RESPONSE_CODE"

    if [ "$RESPONSE_CODE" -eq 200 ]; then
        log_message "Backup script completed successfully."
    else
        log_message "Backup script failed with HTTP code: $RESPONSE_CODE"
        send_email "Backup script failed with HTTP code: $RESPONSE_CODE"
    fi
}

# Run the backup script
run_backup





























# mb
#!/bin/bash

# Define the remote PHP backup script URL
BACKUP_URL="http://remote-server.com/backup.php"

# Define the log file location
LOG_FILE="/path/to/backup_log.txt"

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
    send_email
    exit 1
else
    log_message "Backup script completed successfully."
    exit 0
fi



















































