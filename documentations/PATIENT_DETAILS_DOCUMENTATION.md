# Patient Details Feature Documentation

## Overview
The Patient Details feature allows patients to complete their comprehensive profile information after signing up. This includes personal information, guardian/emergency contact details, medical history, insurance information, and consent forms.

## Files Created/Modified

### 1. Models
- `/src/models/Patient.js` - Mongoose schema for patient details
  - Personal information (DOB, gender, phone, address, occupation, marital status)
  - Guardian information (for minors or emergency contacts)
  - Medical information (physician, allergies, medications, conditions, previous therapy)
  - Insurance information
  - Preferences (contact method, therapist gender preference, session format)
  - Consents (treatment, HIPAA, communication, guardian consent for minors)

### 2. API Routes
- `/src/app/api/patients/route.js` - Main CRUD operations for patient details
  - GET: Retrieve patient details
  - POST: Create new patient details
  - PUT: Update existing patient details
  - DELETE: Soft delete patient details

- `/src/app/api/patients/profile/route.js` - Profile-specific operations
  - GET: Get patient profile with user info
  - PUT: Update patient profile

### 3. Frontend Components
- `/src/app/patient-details/page.jsx` - Multi-step form for patient onboarding
  - Step 1: Personal Information
  - Step 2: Guardian/Emergency Contact
  - Step 3: Medical & Insurance
  - Step 4: Consent & Preferences

### 4. Hooks
- `/src/hooks/usePatientData.js` - Custom hook for managing patient data
  - Fetch patient data
  - Update patient data
  - Handle loading states and errors

## Features

### Form Persistence
- **Auto-save**: Form data is automatically saved to localStorage every 500ms after changes
- **Step Persistence**: Current step is saved and restored on page reload
- **Draft Recovery**: Previous drafts are automatically loaded when returning to the form
- **Visual Indicators**: Shows saving status and last saved timestamp
- **Manual Clear**: Users can manually clear saved form data with confirmation
- **Automatic Cleanup**: Saved data is automatically cleared upon successful form submission

### Multi-Step Form
1. **Personal Information**
   - Date of birth, gender, phone number
   - Complete address information
   - Occupation and marital status

2. **Guardian/Emergency Contact**
   - Automatically detects if user is minor (under 18)
   - Required guardian information for minors
   - Emergency contact information for adults
   - Multiple guardian support

3. **Medical & Insurance**
   - Primary physician details
   - Allergies (with dynamic add/remove)
   - Current medications
   - Medical conditions
   - Previous therapy experience
   - Insurance information (optional)

4. **Consent & Preferences**
   - Communication preferences
   - Therapist gender preference
   - Session format preference
   - Required consent forms (treatment, HIPAA, communication)
   - Guardian consent for minors

### Key Features
- **Age Detection**: Automatically detects if patient is a minor based on date of birth
- **Dynamic Forms**: Add/remove items for allergies, medications, etc.
- **Validation**: Step-by-step validation with error messages
- **Responsive Design**: Mobile-friendly interface
- **Progress Tracking**: Visual progress indicator
- **Data Persistence**: Saves to MongoDB with proper relationships
- **Form Auto-Save**: Automatic localStorage persistence with recovery
- **Draft Management**: Save, load, and clear draft functionality

### Database Structure
```javascript
Patient {
  userId: ObjectId (ref to User)
  clerkId: String (Clerk user ID)
  personalInfo: {
    dateOfBirth: Date
    gender: String
    phoneNumber: String
    address: Object
    occupation: String
    maritalStatus: String
  }
  guardianInfo: {
    isMinor: Boolean
    guardians: Array of guardian objects
  }
  medicalInfo: {
    primaryPhysician: Object
    allergies: Array
    currentMedications: Array
    medicalConditions: Array
    previousTherapy: Object
  }
  insuranceInfo: Object
  preferences: Object
  consents: Object
  status: {
    profileComplete: Boolean
    activePatient: Boolean
    assignedTherapist: ObjectId
  }
}
```

### API Endpoints

#### GET /api/patients
- Returns patient details for authenticated user
- Includes user information
- Indicates if details exist

#### POST /api/patients
- Creates new patient details
- Validates required fields
- Updates user profile completion status

#### PUT /api/patients
- Updates existing patient details
- Creates new record if doesn't exist
- Updates user profile completion status

#### DELETE /api/patients
- Soft deletes patient (sets activePatient: false)
- Updates user profile completion status

### Security
- Authentication required (Clerk)
- Role-based access (patients only)
- Input validation
- Data sanitization
- HIPAA compliance considerations

### Error Handling
- Form validation with user-friendly messages
- API error responses
- Loading states
- Network error handling

## Usage

### For Patients
1. After signup, patients are redirected to `/patient-details`
2. Complete the 4-step form
3. Submit to save profile
4. Redirect to dashboard

### For Developers
```javascript
// Use the custom hook
import { usePatientData } from '@/hooks/usePatientData';

function MyComponent() {
  const { 
    patientData, 
    loading, 
    error, 
    hasDetails, 
    updatePatientData 
  } = usePatientData();

  // Component logic
}
```

## Future Enhancements
- File upload for documents
- Electronic signature for consents
- Integration with calendar for availability
- Notification preferences
- Multi-language support
- Export patient data
- Advanced medical history forms