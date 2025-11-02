import { auth } from '@clerk/nextjs/server';
import { NextResponse } from 'next/server';
import connectDB from '@/lib/mongodb';
import User from '@/models/User';
import Therapist from '@/models/Therapist';

// GET /api/therapists/available - Get all available therapists for patient assignment
export async function GET(req) {
  try {
    const { userId } = await auth(req);
    
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    await connectDB();
    
    // Verify the user is authenticated (can be patient or admin)
    const user = await User.findOne({ clerkId: userId });
    
    if (!user) {
      return NextResponse.json({ error: 'User not found' }, { status: 403 });
    }

    // Get all active therapists
    const therapists = await User.find({ 
      role: 'therapist', 
      isActive: true 
    }).select('_id firstName lastName email');

    console.log(therapists)

    // Get additional therapist info if available
    const therapistList = await Promise.all(therapists.map(async (therapist) => {
      const therapistDetails = await Therapist.findOne({ userId: therapist._id });
      
      return {
        _id: therapistDetails?._id || therapist._id, // Use Therapist document _id, not User _id
        userId: therapist._id, // Store User _id separately
        name: `${therapist.firstName} ${therapist.lastName}`,
        email: therapist.email,
        specializations: therapistDetails?.professionalInfo?.specializations || [],
        acceptingNewPatients: therapistDetails?.preferences?.acceptingNewPatients ?? true,
        yearsOfExperience: therapistDetails?.professionalInfo?.yearsOfExperience || 0,
        verified: therapistDetails?.status?.verified || false
      };
    }));

    console.log(therapistList)
    // Filter only therapists accepting new patients
    // const availableTherapists = therapistList.filter(therapist => 
    //   therapist.acceptingNewPatients && therapist.verified
    // );

    console.log("available therapist: ",therapistList);

    return NextResponse.json({ therapists: therapistList }, { status: 200 });

  } catch (error) {
    console.error('Error fetching available therapists:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}