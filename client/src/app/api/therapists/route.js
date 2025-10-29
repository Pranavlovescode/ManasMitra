import { auth } from '@clerk/nextjs/server';
import { NextResponse } from 'next/server';
import connectDB from '@/lib/mongodb';
import User from '@/models/User';

// GET /api/therapists/patients - Get all patients for a therapist
export async function GET() {
  try {
    const { userId } = auth();
    
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    await connectDB();
    
    // Verify the user is a therapist
    const therapist = await User.findOne({ clerkId: userId, role: 'therapist' });
    
    if (!therapist) {
      return NextResponse.json({ error: 'Access denied. Therapist account required.' }, { status: 403 });
    }

    // For now, return all patients (in a real app, you'd have a relationship model)
    const patients = await User.find({ 
      role: 'patient', 
      isActive: true 
    }).select('firstName lastName email createdAt lastLogin profileComplete');

    return NextResponse.json({ patients });

  } catch (error) {
    console.error('Error fetching patients:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}

// GET /api/therapists/dashboard - Get therapist dashboard data
export async function POST(req) {
  try {
    const { userId } = auth();
    
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    await connectDB();
    
    // Verify the user is a therapist
    const therapist = await User.findOne({ clerkId: userId, role: 'therapist' });
    
    if (!therapist) {
      return NextResponse.json({ error: 'Access denied. Therapist account required.' }, { status: 403 });
    }

    // Get dashboard statistics
    const totalPatients = await User.countDocuments({ role: 'patient', isActive: true });
    const newPatientsThisMonth = await User.countDocuments({
      role: 'patient',
      isActive: true,
      createdAt: { $gte: new Date(new Date().getFullYear(), new Date().getMonth(), 1) }
    });
    
    const activePatients = await User.countDocuments({
      role: 'patient',
      isActive: true,
      lastLogin: { $gte: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000) } // Last 30 days
    });

    return NextResponse.json({
      dashboardData: {
        totalPatients,
        newPatientsThisMonth,
        activePatients,
        therapistInfo: {
          name: therapist.fullName,
          specializations: therapist.specializations,
          yearsOfExperience: therapist.yearsOfExperience,
          acceptingPatients: therapist.acceptingPatients
        }
      }
    });

  } catch (error) {
    console.error('Error fetching dashboard data:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}