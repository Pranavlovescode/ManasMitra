# ManasMitra - Mental Health Journaling App

ManasMitra is a comprehensive mental health journaling platform designed for both patients and therapists. Built with Next.js and integrated with Clerk authentication, it provides a secure and user-friendly environment for mental health support.

## Features

### For Patients
- 🔒 Secure personal journaling
- 📊 Daily mood tracking
- 👩‍⚕️ Therapist connection
- 📈 Progress visualization
- 🔔 Appointment scheduling

### For Therapists
- 👥 Patient management dashboard
- 📊 Progress analytics
- 📅 Appointment scheduling
- 📋 Treatment plan management
- ⚙️ Practice settings

## Tech Stack

- **Frontend**: Next.js 16.0.0 with React 19.2.0
- **Authentication**: Clerk
- **Styling**: Tailwind CSS v4
- **Icons**: Lucide React
- **Deployment**: Vercel (recommended)

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn
- Clerk account

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ManasMitra/client
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up Clerk Authentication**
   
   Create a Clerk application at [clerk.com](https://clerk.com) and get your API keys.

4. **Environment Variables**
   
   Update the `.env.local` file with your Clerk credentials:
   ```env
   NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_your_publishable_key_here
   CLERK_SECRET_KEY=sk_test_your_secret_key_here
   NEXT_PUBLIC_CLERK_SIGN_IN_URL=/sign-in
   NEXT_PUBLIC_CLERK_SIGN_UP_URL=/sign-up
   NEXT_PUBLIC_CLERK_AFTER_SIGN_IN_URL=/dashboard
   NEXT_PUBLIC_CLERK_AFTER_SIGN_UP_URL=/dashboard
   ```

5. **Run the development server**
   ```bash
   npm run dev
   ```

6. **Open your browser**
   
   Navigate to [http://localhost:3000](http://localhost:3000)

## Project Structure

```
client/
├── src/
│   ├── app/
│   │   ├── dashboard/          # Protected dashboard pages
│   │   ├── sign-in/           # Sign-in page with Clerk
│   │   ├── sign-up/           # Sign-up page with Clerk
│   │   ├── globals.css        # Global styles
│   │   ├── layout.js          # Root layout with Clerk provider
│   │   └── page.js            # Landing page
│   ├── components/            # Reusable components
│   └── middleware.js          # Clerk authentication middleware
├── public/                    # Static assets
├── .env.local                 # Environment variables
└── package.json
```

## Authentication Flow

1. **Landing Page**: Users choose between Patient or Therapist role
2. **Sign-up**: Role-specific sign-up with Clerk authentication
3. **Sign-in**: Unified sign-in for all users
4. **Dashboard**: Role-based dashboard after authentication

## User Roles

The app supports two user types:

- **Patient**: Access to journaling, mood tracking, and therapist connection
- **Therapist**: Patient management, analytics, and appointment scheduling

User roles are stored in Clerk's `unsafeMetadata` and used throughout the app for role-based access control.

## Styling

The app uses Tailwind CSS v4 with custom mental health-themed styling:

- Calming color palette (blues, greens, purples)
- Gradient backgrounds and buttons
- Smooth transitions and hover effects
- Responsive design for all devices

## Security Features

- HIPAA compliant design principles
- Clerk authentication with secure session management
- Protected routes using middleware
- Role-based access control
- Secure data handling practices

## Deployment

### Vercel (Recommended)

1. Connect your GitHub repository to Vercel
2. Add environment variables in Vercel dashboard
3. Deploy automatically on push to main branch

### Other Platforms

The app can be deployed to any platform that supports Next.js applications.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## Support

For support and questions:

- Create an issue in the repository
- Contact the development team
- Check the documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This is a mental health application. Please ensure compliance with local healthcare regulations and consider consulting with healthcare professionals when implementing features related to patient care.
