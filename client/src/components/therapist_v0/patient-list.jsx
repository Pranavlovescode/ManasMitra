"use client";

import { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../ui/card";
import { Button } from "../ui/button";
import { Input } from "../ui/input";

export default function PatientList({ therapistId, onSelectPatient }) {
  // Type annotation removed
  const [patients, setPatients] = useState([]); // Type <any[]> removed
  const [filteredPatients, setFilteredPatients] = useState([]); // Type <any[]> removed
  const [searchTerm, setSearchTerm] = useState("");
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchPatients();
  }, [therapistId]);

  useEffect(() => {
    const filtered = patients.filter(
      (p) => {
        const firstName = p.userId?.firstName || '';
        const lastName = p.userId?.lastName || '';
        const email = p.userId?.email || '';
        const searchLower = searchTerm.toLowerCase();
        
        return firstName.toLowerCase().includes(searchLower) ||
               lastName.toLowerCase().includes(searchLower) ||
               email.toLowerCase().includes(searchLower);
      }
    );
    setFilteredPatients(filtered);
  }, [searchTerm, patients]);

  const fetchPatients = async () => {
    try {
      const res = await fetch('/api/therapists/patients');
      
      if (res.ok) {
        const data = await res.json();
        // Extract just the assigned patients or all patients
        setPatients(data.assignedPatients || data.allPatients || []);
      }
    } catch (error) {
      console.error("Failed to fetch patients:", error);
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="text-center text-muted-foreground">
        Loading patients...
      </div>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Your Patients</CardTitle>
        <CardDescription>Manage and monitor your patient list</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Input
          placeholder="Search patients..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="mb-4"
        />

        <div className="space-y-2">
          {filteredPatients.length > 0 ? (
            filteredPatients.map((patient) => (
              <div
                key={patient._id}
                className="flex items-center justify-between p-4 bg-muted rounded-lg hover:bg-muted/80"
              >
                <div>
                  <p className="font-semibold">
                    {patient.userId?.firstName} {patient.userId?.lastName}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {patient.userId?.email}
                  </p>
                </div>
                <Button size="sm" onClick={() => onSelectPatient(patient._id)}>
                  View
                </Button>
              </div>
            ))
          ) : (
            <p className="text-center text-muted-foreground py-8">
              No patients found
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
