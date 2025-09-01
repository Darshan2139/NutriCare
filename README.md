# NutriCare - AI-Powered Maternal Nutrition Platform

<div align="center">
  <img src="public/favicon.png" alt="NutriCare Logo" width="120" height="120" />
  <h1>🍎 NutriCare</h1>
  <p><strong>AI-Powered Maternal Nutrition & Health Management Platform</strong></p>
  
  [![React](https://img.shields.io/badge/React-18.3.1-blue.svg)](https://reactjs.org/)
  [![TypeScript](https://img.shields.io/badge/TypeScript-5.5.3-blue.svg)](https://www.typescriptlang.org/)
  [![Node.js](https://img.shields.io/badge/Node.js-Express-green.svg)](https://nodejs.org/)
  [![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green.svg)](https://www.mongodb.com/)
  [![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4.11-38B2AC.svg)](https://tailwindcss.com/)
  [![Gemini AI](https://img.shields.io/badge/Gemini-AI-orange.svg)](https://ai.google.dev/)
</div>

## 🌟 Overview

NutriCare is a comprehensive maternal nutrition and health management platform that leverages AI to provide personalized diet plans, health tracking, and medical guidance for pregnant women. The platform combines modern web technologies with Google's Gemini AI to deliver intelligent, data-driven nutrition recommendations.

## 👥 Team DATA DREAMERS

**A passionate team dedicated to revolutionizing maternal healthcare through technology and AI.**

### Team Members:
- **Darshan Kachhiya** - Full Stack Development & AI Integration
- **Dharmit Fadadu** - Backend Development & Database Architecture  
- **Shreya Ghelani** - Frontend Development & UI/UX Design
- **Kedar Kotahri** - System Architecture & DevOps

## 📸 Application Screenshots

<div align="center">
  <h3>🎯 Key Features Showcase</h3>
  
  <table>
    <tr>
      <td align="center">
        <strong>Hero Section</strong><br/>
        <img src="Nutricare images/1.png" alt="NutriCare Hero Section" width="200"/>
        <br/>Personalized nutrition landing page
      </td>
      <td align="center">
        <strong>Dashboard</strong><br/>
        <img src="Nutricare images/7.png" alt="NutriCare Dashboard" width="200"/>
        <br/>Comprehensive health analytics
      </td>
    </tr>
    <tr>
      <td align="center">
        <strong>AI Processing</strong><br/>
        <img src="Nutricare images/16.png" alt="AI Diet Plan Generation" width="200"/>
        <br/>Real-time AI analysis
      </td>
      <td align="center">
        <strong>Diet Plan Results</strong><br/>
        <img src="Nutricare images/18.png" alt="Diet Plan Overview" width="200"/>
        <br/>Personalized nutrition insights
      </td>
    </tr>
    <tr>
      <td align="center">
        <strong>Weekly Meal Plan</strong><br/>
        <img src="Nutricare images/19.png" alt="Weekly Meal Plan" width="200"/>
        <br/>Detailed 7-day nutrition plan
      </td>
      <td align="center">
        <strong>Exercise Videos</strong><br/>
        <img src="Nutricare images/25.png" alt="Prenatal Exercise Videos" width="200"/>
        <br/>Safe prenatal workouts
      </td>
    </tr>
    <tr>
      <td align="center">
        <strong>AI Chatbot</strong><br/>
        <img src="Nutricare images/27.png" alt="AI Nutrition Assistant" width="200"/>
        <br/>24/7 nutrition guidance
      </td>
      <td align="center">
        <strong>Emergency Help</strong><br/>
        <img src="Nutricare images/30.png" alt="Emergency Services" width="200"/>
        <br/>Hospital finder & emergency contacts
      </td>
    </tr>
    <tr>
      <td align="center" colspan="2">
        <strong>User Profile</strong><br/>
        <img src="Nutricare images/34.png" alt="User Profile Management" width="200"/>
        <br/>Personal information & health data management
      </td>
    </tr>
  </table>
</div>

## ✨ Key Features

### 🤖 AI-Powered Diet Plan Generation
- **Personalized Nutrition Plans**: AI-generated diet plans based on comprehensive health data
- **Multi-language Support**: Diet plans available in English and Gujarati
- **Cultural Adaptation**: Respects dietary restrictions and cultural preferences
- **Real-time Analysis**: Instant health scoring and nutritional insights

### 📊 Comprehensive Health Dashboard
- **Health Analytics**: Visual representation of health metrics and trends
- **Progress Tracking**: Monitor nutrition goals and health improvements
- **Meal Completion Tracking**: Track daily meal adherence
- **Health Score Monitoring**: Overall wellness scoring system

### 🏥 Medical Integration
- **Hospital Finder**: Locate nearby hospitals and medical facilities
- **Emergency Services**: Quick access to emergency contact information
- **Health Report Upload**: Upload and analyze medical reports
- **Lab Value Tracking**: Monitor blood work and vital signs

### 💬 AI Chatbot Assistant
- **24/7 Health Support**: AI-powered health consultation
- **Nutritional Guidance**: Real-time dietary advice
- **Medical Information**: Evidence-based health information
- **Conversation History**: Persistent chat history for continuity

### 📱 Modern User Experience
- **Responsive Design**: Optimized for desktop, tablet, and mobile
- **Progressive Web App**: Offline capabilities and app-like experience
- **Real-time Updates**: Live data synchronization
- **Intuitive Navigation**: User-friendly interface with smooth animations

## 🛠️ Tech Stack

### Frontend
- **React 18** - Modern UI framework with hooks
- **TypeScript** - Type-safe development
- **Vite** - Fast build tool and dev server
- **React Router 6** - Client-side routing
- **TailwindCSS 3** - Utility-first CSS framework
- **Radix UI** - Accessible component primitives
- **Framer Motion** - Smooth animations and transitions
- **Lucide React** - Beautiful icon library

### Backend
- **Node.js** - JavaScript runtime
- **Express.js** - Web application framework
- **MongoDB** - NoSQL database with Mongoose ODM
- **JWT** - Authentication and authorization
- **bcryptjs** - Password hashing
- **Cloudinary** - Image upload and management

### AI & External Services
- **Google Gemini AI** - Advanced AI model for diet plan generation
- **Google Maps API** - Hospital location services
- **PDF Generation** - Diet plan export functionality

### Development Tools
- **Vitest** - Unit testing framework
- **Prettier** - Code formatting
- **ESLint** - Code linting
- **SWC** - Fast TypeScript/JavaScript compiler

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ 
- npm or yarn
- MongoDB Atlas account
- Google Cloud account (for Gemini AI)
- Google Maps API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd NutriCare-30
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Environment Setup**
   Create a `.env` file in the root directory:
   ```env
   # Server Configuration
   PORT=8080
   NODE_ENV=development

   # Database
   MONGO_URI=mongodb+srv://your-username:your-password@your-cluster.mongodb.net/nutricare

   # JWT Configuration
   JWT_SECRET=your-super-secret-jwt-key-here
   JWT_EXPIRE=7d

   # Google Gemini API
   GEMINI_API_KEY=your-gemini-api-key-here

   # Google Maps API
   GOOGLE_MAPS_API_KEY=your-google-maps-api-key

   # Cloudinary (for image uploads)
   CLOUDINARY_CLOUD_NAME=your-cloud-name
   CLOUDINARY_API_KEY=your-api-key
   CLOUDINARY_API_SECRET=your-api-secret
   ```

4. **Start Development Server**
   ```bash
   npm run dev
   ```

5. **Open Application**
   Navigate to `http://localhost:8080`


## 🔧 Available Scripts

```bash
# Development
npm run dev              # Start development server (client + server)
npm run typecheck        # TypeScript validation

# Building
npm run build           # Build for production
npm run build:client    # Build only frontend
npm run build:server    # Build only backend

# Production
npm run start           # Start production server

# Testing
npm test               # Run Vitest tests

# Code Quality
npm run format.fix      # Format code with Prettier
```

## 🎯 Core Features Deep Dive

### AI Diet Plan Generation

The platform uses Google's Gemini AI to generate personalized diet plans based on comprehensive health data:

**Input Data:**
- Demographics (age, height, weight, BMI, pregnancy stage)
- Medical data (hemoglobin, blood sugar, blood pressure)
- Lab values (vitamin levels, calcium, iron)
- Dietary preferences and restrictions
- Lifestyle factors (activity level, sleep, water intake)

**Output:**
- Overall health score (0-100)
- Personalized recommendations
- 7-day detailed meal plan
- Nutritional insights and priorities
- Supplement recommendations
- Foods to avoid/limit

### Health Dashboard

Comprehensive analytics dashboard featuring:
- **Health Metrics Visualization**: Charts and graphs for key health indicators
- **Progress Tracking**: Monitor improvements over time
- **Meal Completion**: Track daily nutrition adherence
- **Goal Setting**: Set and monitor nutrition targets
- **Report Analysis**: Upload and analyze medical reports

### Hospital Finder

Emergency services integration:
- **Nearby Hospitals**: Find closest medical facilities
- **Hospital Details**: Contact information and services
- **Emergency Contacts**: Quick access to emergency numbers
- **Geolocation**: Automatic location detection

## 🔐 Authentication & Security

- **JWT-based Authentication**: Secure token-based authentication
- **Password Hashing**: bcryptjs for secure password storage
- **Protected Routes**: Role-based access control
- **Input Validation**: Zod schema validation
- **CORS Configuration**: Secure cross-origin requests

## 📊 Database Schema

### User Model
```typescript
{
  name: string,
  email: string,
  password: string (hashed),
  profile: {
    age: number,
    height: number,
    weight: number,
    pregnancyStage: string,
    dietaryRestrictions: string[]
  },
  createdAt: Date,
  updatedAt: Date
}
```

### Health Entry Model
```typescript
{
  userId: ObjectId,
  healthData: {
    hemoglobin: number,
    bloodSugar: number,
    bloodPressure: string,
    labValues: object,
    medicalHistory: string[]
  },
  createdAt: Date
}
```

### Diet Plan Model
```typescript
{
  userId: ObjectId,
  healthScore: number,
  recommendations: string[],
  weeklyPlan: object,
  supplements: object[],
  restrictions: string[],
  generatedAt: Date
}
```

## 🌐 API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `POST /api/auth/logout` - User logout
- `GET /api/auth/profile` - Get user profile

### Health Management
- `POST /api/health/entry` - Create health entry
- `GET /api/health/entries` - Get health history
- `POST /api/health/upload` - Upload medical reports

### Diet Plans
- `POST /api/plans/generate` - Generate AI diet plan
- `GET /api/plans/history` - Get plan history
- `POST /api/plans/complete-meal` - Mark meal as completed

### Analytics
- `GET /api/analytics/dashboard` - Get dashboard analytics
- `GET /api/analytics/health-trends` - Get health trends

### Emergency Services
- `GET /api/hospitals/nearby` - Find nearby hospitals
- `GET /api/hospitals/:id` - Get hospital details

### Chatbot
- `POST /api/chatbot/message` - Send chat message
- `GET /api/chatbot/history/:userId` - Get chat history

## 🚀 Deployment

### Production Build
```bash
npm run build
npm start
```

### Environment Variables for Production
```env
NODE_ENV=production
PORT=3000
MONGO_URI=your-production-mongodb-uri
JWT_SECRET=your-production-jwt-secret
GEMINI_API_KEY=your-production-gemini-key
GOOGLE_MAPS_API_KEY=your-production-maps-key
CLOUDINARY_CLOUD_NAME=your-production-cloudinary-name
CLOUDINARY_API_KEY=your-production-cloudinary-key
CLOUDINARY_API_SECRET=your-production-cloudinary-secret
```
## 🧪 Testing

```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run specific test file
npm test -- Dashboard.test.tsx
```
## 🙏 Acknowledgments

- **Google Gemini AI** - For providing the AI capabilities
- **Radix UI** - For the excellent component library
- **TailwindCSS** - For the utility-first CSS framework
- **React Team** - For the amazing React framework
- **Express.js** - For the robust backend framework

## 🔄 Version History

- **v1.0.0** - Initial release with core features
- **v1.1.0** - Added AI chatbot and hospital finder
- **v1.2.0** - Enhanced dashboard analytics
- **v1.3.0** - Multi-language support and PDF export

---

<div align="center">
  <p>Made with ❤️ for maternal health and nutrition</p>
  <p>Built with modern web technologies and AI</p>
</div>
