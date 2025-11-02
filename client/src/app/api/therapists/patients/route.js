import { getAuth, currentUser } from '@clerk/nextjs/server';
import { NextResponse } from 'next/server';
import connectDB from '../../../../lib/mongodb';
import User from '../../../../models/User';
import Patient from '../../../../models/Patient';

// GET /api/therapists/patients - Get all patients and assigned patients for a therapist
export async function GET(request) {
  try {
    const auth = getAuth(request);
    const { userId } = auth;

    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    await connectDB();
    console.log('MongoDB connected successfully');

    // Find the therapist or create if doesn't exist
    let therapist = await User.findOne({ clerkId: userId });
    console.log('Therapist lookup:', therapist ? `${therapist.firstName} ${therapist.lastName}` : 'Not found');

    if (!therapist) {
      // Auto-create therapist from Clerk data
      console.log('Therapist not found in DB, creating from Clerk data...');
      const clerkUser = await currentUser();
      
      if (!clerkUser) {
        return NextResponse.json({ error: 'User not found in Clerk' }, { status: 404 });
      }

      therapist = User.createSafeUser({
        clerkId: clerkUser.id,
        email: clerkUser.emailAddresses[0]?.emailAddress,
        firstName: clerkUser.firstName || 'User',
        lastName: clerkUser.lastName || 'Unknown', // Provide default if missing
        role: clerkUser.unsafeMetadata?.role || 'therapist',
        profileImage: clerkUser.imageUrl,
        profileComplete: false,
        isActive: true
      });

      await therapist.save();
      console.log('Therapist created successfully:', therapist.email);
    }

    if (therapist.role !== 'therapist') {
      console.log('User role is not therapist:', therapist.role);
      return NextResponse.json({ error: 'Access denied. Only therapists can access this endpoint.' }, { status: 403 });
    }

    // Get all patients - no filtering by assignment
    console.log('Fetching all active patients...');
    let allPatients = await Patient.find({ 'status.activePatient': true })
      .populate('userId', 'firstName lastName email profileImage')
      .sort({ createdAt: -1 });

    console.log(`Found ${allPatients.length} active patients in database`);
    
    // Also check total patients without filter
    const totalPatientsCount = await Patient.countDocuments({});
    console.log(`Total patients in database (including inactive): ${totalPatientsCount}`);
    
    // If no active patients found, get all patients regardless of status
    if (allPatients.length === 0) {
      console.log('No active patients found, fetching all patients...');
      allPatients = await Patient.find({})
        .populate('userId', 'firstName lastName email profileImage')
        .sort({ createdAt: -1 });
      console.log(`Found ${allPatients.length} total patients`);
    }

    // For now, treat one patient as assigned (first patient) and rest as unassigned
    const assignedPatients = allPatients.length > 0 ? [allPatients[0]] : [];
    const unassignedPatients = allPatients.slice(1);

    return NextResponse.json({
      allPatients: allPatients.map(patient => ({
        _id: patient._id,
        userId: patient.userId,
        personalInfo: patient.personalInfo,
        guardianInfo: patient.guardianInfo,
        medicalInfo: patient.medicalInfo,
        preferences: patient.preferences,
        status: patient.status,
        age: patient.age,
        createdAt: patient.createdAt,
        isAssigned: patient._id.toString() === allPatients[0]?._id?.toString(),
      })),
      assignedPatients: assignedPatients.map(patient => ({
        _id: patient._id,
        userId: patient.userId,
        personalInfo: patient.personalInfo,
        medicalInfo: patient.medicalInfo,
        preferences: patient.preferences,
        guardianInfo: patient.guardianInfo,
        status: patient.status,
        age: patient.age,
        createdAt: patient.createdAt,
      })),
      unassignedPatients: unassignedPatients.map(patient => ({
        _id: patient._id,
        userId: patient.userId,
        personalInfo: patient.personalInfo,
        guardianInfo: patient.guardianInfo,
        medicalInfo: patient.medicalInfo,
        preferences: patient.preferences,
        status: patient.status,
        age: patient.age,
        createdAt: patient.createdAt,
      })),
      stats: {
        totalPatients: allPatients.length,
        assignedCount: assignedPatients.length,
        unassignedCount: unassignedPatients.length,
      }
    });
  } catch (error) {
    console.error('Error fetching patients:', error);
    console.error('Error stack:', error.stack);
    return NextResponse.json({ 
      error: 'Internal Server Error',
      message: error.message,
      details: process.env.NODE_ENV === 'development' ? error.stack : undefined
    }, { status: 500 });
  }
}
