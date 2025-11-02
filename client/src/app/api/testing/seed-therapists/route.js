import { NextResponse } from 'next/server';
import connectDB from '@/lib/mongodb';
import User from '@/models/User';
import Therapist from '@/models/Therapist';

// POST /api/testing/seed-therapists - Create test therapists (for development only)
export async function POST() {
  try {
    // Only allow in development environment
    if (process.env.NODE_ENV !== 'development') {
      return NextResponse.json({ error: 'This endpoint is only available in development' }, { status: 403 });
    }

    await connectDB();

    const testTherapists = [
      {
        user: {
          clerkId: 'test_therapist_1',
          email: 'dr.smith@mentalhealth.com',
          firstName: 'Dr. Sarah',
          lastName: 'Smith',
          role: 'therapist',
          profileComplete: true,
          isActive: true
        },
        therapist: {
          professionalInfo: {
            licenseNumber: 'LIC123456',
            licenseState: 'CA',
            yearsOfExperience: 10,
            specializations: ['Anxiety', 'Depression', 'CBT'],
            therapyApproaches: ['Cognitive Behavioral Therapy', 'Mindfulness-Based Therapy'],
            languages: ['English', 'Spanish']
          },
          preferences: {
            sessionFormats: ['virtual', 'in-person'],
            acceptingNewPatients: true
          },
          status: {
            profileComplete: true,
            isActive: true,
            verified: true
          }
        }
      },
      {
        user: {
          clerkId: 'test_therapist_2',
          email: 'dr.johnson@mentalhealth.com',
          firstName: 'Dr. Michael',
          lastName: 'Johnson',
          role: 'therapist',
          profileComplete: true,
          isActive: true
        },
        therapist: {
          professionalInfo: {
            licenseNumber: 'LIC789012',
            licenseState: 'NY',
            yearsOfExperience: 15,
            specializations: ['PTSD', 'Trauma', 'Family Therapy'],
            therapyApproaches: ['EMDR', 'Family Systems Therapy'],
            languages: ['English']
          },
          preferences: {
            sessionFormats: ['virtual'],
            acceptingNewPatients: true
          },
          status: {
            profileComplete: true,
            isActive: true,
            verified: true
          }
        }
      },
      {
        user: {
          clerkId: 'test_therapist_3',
          email: 'dr.williams@mentalhealth.com',
          firstName: 'Dr. Emily',
          lastName: 'Williams',
          role: 'therapist',
          profileComplete: true,
          isActive: true
        },
        therapist: {
          professionalInfo: {
            licenseNumber: 'LIC345678',
            licenseState: 'TX',
            yearsOfExperience: 8,
            specializations: ['Adolescent Therapy', 'Eating Disorders'],
            therapyApproaches: ['Dialectical Behavior Therapy', 'Solution-Focused Therapy'],
            languages: ['English', 'French']
          },
          preferences: {
            sessionFormats: ['in-person', 'virtual'],
            acceptingNewPatients: true
          },
          status: {
            profileComplete: true,
            isActive: true,
            verified: true
          }
        }
      }
    ];

    const createdTherapists = [];

    for (const therapistData of testTherapists) {
      // Check if user already exists
      let user = await User.findOne({ clerkId: therapistData.user.clerkId });
      
      if (!user) {
        user = new User(therapistData.user);
        await user.save();
      }

      // Check if therapist profile already exists
      let therapist = await Therapist.findOne({ clerkId: therapistData.user.clerkId });
      
      if (!therapist) {
        therapist = new Therapist({
          userId: user._id,
          clerkId: therapistData.user.clerkId,
          ...therapistData.therapist
        });
        await therapist.save();
      }

      createdTherapists.push({
        user: user,
        therapist: therapist
      });
    }

    return NextResponse.json({ 
      message: 'Test therapists created/verified successfully',
      therapists: createdTherapists.length
    });

  } catch (error) {
    console.error('Error creating test therapists:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}