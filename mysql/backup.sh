#!/bin/bash

# MySQL 데이터베이스 정보
DB_NAME="CJD"  # 데이터베이스 이름
DB_USER="pdc"       # MySQL 사용자 이름
DB_PASS="1234"       # MySQL 비밀번호
DUMP_FILE="backup.sql"        # 백업 파일 이름

# 현재 날짜를 추가하여 백업 파일명 구성 (선택 사항)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="backup_${TIMESTAMP}.sql"

# MySQL 데이터 덤프 생성
echo "Backing up MySQL database: $DB_NAME"
mysqldump -u $DB_USER -p$DB_PASS $DB_NAME > $DUMP_FILE

# Git 커밋 및 푸시
echo "Committing and pushing changes to Git..."
git add $DUMP_FILE
git commit -m "Backup on $TIMESTAMP"
git push origin main

echo "Backup and Git synchronization complete!"
