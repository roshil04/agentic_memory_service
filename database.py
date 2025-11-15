from google.adk.sessions import DatabaseSessionService

# Using PostgreSQL
db_url = "postgresql://postgres:newpassword@localhost:5432/postgres"
session_service = DatabaseSessionService(db_url=db_url)

#sudo -i -u postgres
#psql