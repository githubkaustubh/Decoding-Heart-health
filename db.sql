CREATE DATABASE IF NOT EXISTS hospital;

USE hospital;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    password VARCHAR(255) NOT NULL,
    role ENUM('doctor', 'patient') NOT NULL
);
