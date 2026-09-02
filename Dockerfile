FROM openjdk:17-jdk-slim
WORKDIR /app
# Copy the fat jar built by Maven
COPY backend/target/*.jar app.jar
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "/app/app.jar"]