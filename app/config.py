"""
Application configuration
"""
import os

class Config:
    """Base configuration"""
    DEBUG = False
    TESTING = False
    
class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    
class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    
class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
